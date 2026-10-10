"""Construction-only contracts for Freebase's birth-ge-1920 v3 windows."""

from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from DBpedia.audit_sliding_window_artifacts import interval_mask
from Freebase.configure_sliding_windows import configurations
from Freebase.prepare_sliding_alive2026 import (
    RULE,
    main as construct_main,
    topology_comparison,
    validate_missing_deaths,
    validate_protocol,
)
from Freebase.sliding_multi_gpu import estimate_window_nodes
from Freebase.grouped_pipeline import Pipeline, parse_args as pipeline_args
from data import birth_cohort_artifacts as artifacts
from data.extended import make_numeric_features
from tests import test_birth_cohort_induced_artifacts as induced_fixture
from training.life_periods import life_period_membership, load_life_period_config


ROOT = Path(__file__).resolve().parents[1]
OLD_PIPELINE = ROOT / "config/freebase_grouped_sliding_1900_2000_pm20_step1_v2.json"
OLD_PERIODS = ROOT / "config/freebase_life_windows_1900_2000_pm20_step1_v2.json"
NEW_PIPELINE = ROOT / "config/freebase_grouped_sliding_1900_2000_pm20_step1_born_ge1920_alive2026_v3.json"
NEW_PERIODS = ROOT / "config/freebase_life_windows_1900_2000_pm20_step1_born_ge1920_alive2026_v3.json"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def alive_period_payload(start=1900, end=2000, half_window=20, step=1):
    generated = configurations(
        start,
        end,
        half_window,
        step,
        assume_alive_through=2026,
        born_on_or_after=1920,
    )
    return next(value for path, value in generated.items() if "life_windows" in path.name)


class FreebaseAlive2026Tests(unittest.TestCase):
    def test_inclusive_1920_boundary_and_other_date_policies(self):
        payload = alive_period_payload(1900, 1900, 20, 1)
        payload["periods"] = [{"id": "point_2026", "label": "2026", "start": 2026, "end": 2026}]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "periods.json"
            write_json(config_path, payload)
            config = load_life_period_config(config_path)
            nodes = pd.DataFrame({
                "birth_year": [1920, 1919, 1920, None, None, 2000, 2027],
                "death_year": [None, None, 1940, None, 2026, 1999, None],
            })
            original = nodes.copy(deep=True)
            memberships, audit = life_period_membership(nodes, config)

        self.assertEqual(memberships["point_2026"].tolist(), [True, False, False, False, True, False, False])
        self.assertTrue(nodes.equals(original))
        self.assertEqual(audit.life_period_death_year_imputed.tolist(), [True, False, False, False, False, False, True])
        self.assertTrue(pd.isna(audit.death_year_numeric.iloc[0]))
        self.assertEqual(audit.life_period_effective_death_year.iloc[0], 2026)
        self.assertTrue(pd.isna(audit.life_period_effective_death_year.iloc[1]))
        self.assertEqual(audit.life_period_effective_death_year.iloc[2], 1940)
        self.assertEqual(audit.life_interval_status.iloc[3], "missing_birth_and_death")
        self.assertEqual(audit.life_interval_status.iloc[4], "death_endpoint_only")
        self.assertEqual(audit.life_interval_status.iloc[5], "death_before_birth")
        self.assertEqual(audit.life_interval_status.iloc[6], "death_before_birth")

    def test_generator_keeps_v2_configs_byte_semantics_and_separates_v3_roots(self):
        old = configurations()
        for path, payload in old.items():
            with self.subTest(path=path):
                self.assertEqual(json.loads((ROOT / path).read_text(encoding="utf-8")), payload)

        new = configurations(assume_alive_through=2026, born_on_or_after=1920)
        generated_periods = next(value for path, value in new.items() if "life_windows" in path.name)
        generated_pipeline = next(value for path, value in new.items() if "grouped_sliding" in path.name)
        self.assertEqual(generated_periods, json.loads(NEW_PERIODS.read_text(encoding="utf-8")))
        self.assertEqual(generated_pipeline, json.loads(NEW_PIPELINE.read_text(encoding="utf-8")))
        self.assertEqual(len(generated_periods["periods"]), 101)
        self.assertEqual(generated_periods["periods"][0]["start"], 1880)
        self.assertEqual(generated_periods["periods"][0]["end"], 1920)
        self.assertEqual(generated_periods["periods"][-1]["start"], 1980)
        self.assertEqual(generated_periods["periods"][-1]["end"], 2020)
        self.assertEqual(generated_periods["birth_only_alive_assumption"], RULE)
        old_config = json.loads(OLD_PIPELINE.read_text(encoding="utf-8"))
        for key in ("period_root", "relation_root", "model_root", "graphmask_root"):
            self.assertNotEqual(generated_pipeline[key], old_config[key])
        self.assertEqual(generated_pipeline["source_data"], old_config["source_data"])

    def test_estimator_matches_membership_for_assumed_and_partial_dates(self):
        payload = alive_period_payload(2024, 2026, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "periods.json"
            write_json(config_path, payload)
            period_config = load_life_period_config(config_path)
            source = root / "source"
            source.mkdir()
            node_rows = [
                {"birth_year": "1919", "death_year": ""},
                {"birth_year": "1920", "death_year": ""},
                {"birth_year": "2000", "death_year": "2020"},
                {"birth_year": "", "death_year": ""},
                {"birth_year": "", "death_year": "2025"},
                {"birth_year": "1920", "death_year": "1940"},
                {"birth_year": "2000", "death_year": "1999"},
                {"birth_year": "2027", "death_year": ""},
            ]
            write_csv(source / "nodes.csv", ("birth_year", "death_year"), node_rows)
            nodes = pd.DataFrame(node_rows)
            pipeline = SimpleNamespace(
                path=lambda value: Path(value),
                config={"source_data": str(source / "graph_data.pt")},
                periods={p.identifier: {"id": p.identifier, "start": p.start, "end": p.end}
                         for p in period_config.periods},
                contexts=list(period_config.identifiers),
                period_config=json.loads(config_path.read_text(encoding="utf-8")),
            )
            estimated = estimate_window_nodes(pipeline)
            memberships, _ = life_period_membership(nodes, period_config)

        self.assertEqual(estimated, {key: int(mask.sum()) for key, mask in memberships.items()})

    def test_missing_death_candidates_with_conflict_or_future_status_are_rejected(self):
        for bad_status in ("conflicting_years", "future_year"):
            with self.subTest(death_status=bad_status), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                with (source / "date_audit.tsv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("person_name", "death_status"), delimiter="\t")
                    writer.writeheader()
                    writer.writerows((
                        {"person_name": "candidate", "death_status": bad_status},
                        {"person_name": "older", "death_status": "missing"},
                    ))
                write_csv(source / "nodes.csv", ("node_id", "birth_year", "death_year"), (
                    {"node_id": "candidate", "birth_year": "1920", "death_year": ""},
                    {"node_id": "older", "birth_year": "1919", "death_year": ""},
                ))
                pipeline = SimpleNamespace(path=lambda value: Path(value), config={"source_data": str(source / "graph_data.pt")})
                with self.assertRaisesRegex(ValueError, bad_status):
                    validate_missing_deaths(pipeline)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with (source / "date_audit.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("person_name", "death_status"), delimiter="\t")
                writer.writeheader()
                writer.writerows((
                    {"person_name": "candidate", "death_status": "missing"},
                    {"person_name": "older", "death_status": "missing"},
                    {"person_name": "observed", "death_status": "observed"},
                ))
            write_csv(source / "nodes.csv", ("node_id", "birth_year", "death_year"), (
                {"node_id": "candidate", "birth_year": "1920", "death_year": ""},
                {"node_id": "older", "birth_year": "1919", "death_year": ""},
                {"node_id": "observed", "birth_year": "1930", "death_year": "2020"},
            ))
            pipeline = SimpleNamespace(path=lambda value: Path(value), config={"source_data": str(source / "graph_data.pt")})
            self.assertEqual(validate_missing_deaths(pipeline), 1)

    def test_induced_artifact_records_effective_death_but_rebuilds_temporal_features_from_raw_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data = induced_fixture.PeriodInducedArtifactTests()._write_source_artifact(root)
            nodes_path = source_data.parent / "nodes.csv"
            nodes = pd.read_csv(nodes_path)
            nodes["birth_year"] = [1920, 1990, 1985, 1930, 1990, 1980]
            nodes["death_year"] = [None, 2020, 2000, 2020, 2000, 2010]
            nodes.to_csv(nodes_path, index=False)
            period_payload = alive_period_payload(2000, 2000, 20, 1)
            config_path = root / "alive_periods.json"
            write_json(config_path, period_payload)
            config = load_life_period_config(config_path)

            report = artifacts.build_period_induced_artifact(
                source_data, root / "periods", config, "center_2000", split_seed=42
            )
            artifact = root / "periods/center_2000"
            saved_nodes = pd.read_csv(artifact / "nodes.csv")
            bundle = torch.load(artifact / "graph_data.pt", map_location="cpu", weights_only=False)
            graph = bundle["data"]
            imputed_row = saved_nodes.index[saved_nodes.node_id == "Q0"][0]

            self.assertEqual(report["period_induced_artifact"]["life_date_imputation"]["rule"], RULE)
            self.assertTrue(pd.isna(saved_nodes.loc[imputed_row, "death_year"]))
            self.assertTrue(saved_nodes.loc[imputed_row, "life_period_death_year_imputed"])
            self.assertEqual(saved_nodes.loc[imputed_row, "life_period_effective_death_year"], 2026)
            self.assertEqual(float(graph.temporal[imputed_row, 4]), 1.0)
            self.assertEqual(float(graph.temporal[imputed_row, 5]), 1.0)
            expected_temporal = make_numeric_features(
                saved_nodes,
                saved_nodes.loc[graph.train_mask.numpy(), "node_id"],
            )
            np.testing.assert_allclose(graph.temporal.numpy(), expected_temporal)

    def test_topology_metrics_compare_old_and_new_masks_on_the_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            write_csv(source / "nodes.csv", ("birth_year", "death_year"), (
                {"birth_year": "1920", "death_year": ""},
                {"birth_year": "1980", "death_year": "2020"},
                {"birth_year": "1990", "death_year": "2020"},
                {"birth_year": "1919", "death_year": ""},
            ))
            pd.DataFrame({"source_id": [0, 1], "target_id": [1, 0]}).to_csv(source / "edges.csv", index=False)
            pipeline = SimpleNamespace(
                path=lambda value: Path(value),
                config={"source_data": str(source / "graph_data.pt")},
                periods={"center_2000": {"id": "center_2000", "start": 1980, "end": 2020}},
                contexts=["center_2000"],
                period_config={"birth_only_alive_assumption": RULE},
            )
            row = topology_comparison(pipeline)[0]

        self.assertEqual((row["old_nodes"], row["new_nodes"]), (2, 3))
        self.assertEqual((row["old_directed_edges"], row["new_directed_edges"]), (0, 2))
        self.assertEqual((row["old_isolated_nodes"], row["new_isolated_nodes"]), (2, 1))
        self.assertEqual(row["added_nodes"], 1)
        self.assertEqual(row["previously_isolated_now_connected"], 1)
        self.assertEqual(row["selected_imputed_nodes"], 1)

    def test_plan_has_no_training_probe_or_report_commands(self):
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as output:
            code = construct_main(["plan", "--config", str(NEW_PIPELINE), "--periods", "center_1900"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("[prepare]", text)
        self.assertNotIn("[train]", text)
        self.assertNotIn("[probe]", text)
        self.assertNotIn("[report]", text)
        self.assertNotIn("graphmask-train", text)
        self.assertNotIn("graphmask-report", text)
        self.assertNotIn("RGCN + GraphMask jobs", text)

    def test_constructor_rejects_a_changed_window_grid(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            pipeline = Pipeline(pipeline_args(["plan", "prepare", "--config", str(NEW_PIPELINE), "--device", "cpu"]))
        validate_protocol(pipeline)
        pipeline.period_config["periods"][0]["start"] = 1881
        with self.assertRaisesRegex(ValueError, "101 centers"):
            validate_protocol(pipeline)

    def test_constructor_orchestration_prepares_then_only_collapses_without_gpu_preflight(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in ("graph_data.pt", "nodes.csv", "edges.csv"):
                (source / name).write_bytes(b"fixture\n")
            period_root = root / "periods"
            config_path = root / "pipeline.json"
            config = {
                "source_data": str(source / "graph_data.pt"),
                "period_root": str(period_root),
                "relation_root": str(root / "relations"),
                "model_root": str(root / "models"),
                "graphmask_root": str(root / "graphmask"),
                "prepare": {"split_seed": 42, "train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
                "train": {"epochs": 50},
                "graphmask_train": {"epochs_per_layer": 3},
                "graphmask_report": {"split": "test"},
                "seed": 42,
            }
            write_json(config_path, config)
            fake_pipeline = SimpleNamespace(
                config=config,
                config_path=config_path,
                period_config=alive_period_payload(),
                periods={"center_1900": {"id": "center_1900", "start": 1880, "end": 1920}},
                contexts=["center_1900"],
                representations=["multi_group"],
                args=SimpleNamespace(stage="prepare"),
                failures=[],
                path=lambda value: Path(value),
                run=mock.Mock(return_value=0),
                gpu_preflight=mock.Mock(side_effect=AssertionError("constructor must not initialize a GPU")),
            )
            topology_rows = [{
                "context": "center_1900", "old_nodes": 1, "new_nodes": 2,
                "old_isolated_nodes": 1, "new_isolated_nodes": 0,
                "old_isolated_fraction": 1.0, "new_isolated_fraction": 0.0,
            }]
            audit_result = {"passed": True, "windows": [], "checks": [], "group_counts": []}
            with mock.patch("Freebase.prepare_sliding_alive2026.Pipeline", return_value=fake_pipeline), \
                    mock.patch("Freebase.prepare_sliding_alive2026.output_locks", return_value=nullcontext()), \
                    mock.patch("Freebase.prepare_sliding_alive2026.prepare_shared_source"), \
                    mock.patch("Freebase.prepare_sliding_alive2026.validate_missing_deaths", return_value=1), \
                    mock.patch("Freebase.prepare_sliding_alive2026.GroupedPipeline.prepare"), \
                    mock.patch("Freebase.prepare_sliding_alive2026.code_version", return_value={"git_commit": "test"}), \
                    mock.patch("DBpedia.audit_sliding_window_artifacts.audit", return_value=(fake_pipeline, audit_result)), \
                    mock.patch("Freebase.prepare_sliding_alive2026.topology_comparison", return_value=topology_rows), \
                    redirect_stdout(io.StringIO()):
                result = construct_main(["run", "--config", str(config_path), "--periods", "center_1900"])

            self.assertEqual(result, 0)
            self.assertEqual(fake_pipeline.args.stage, "collapse")
            fake_pipeline.run.assert_called_once_with(summarize=False)
            fake_pipeline.gpu_preflight.assert_not_called()
            manifest = next(period_root.glob("construction_runs/*/manifest.json"))
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["training_status"], "not_started")
            self.assertEqual(record["training_budget_executed"], 0)

    def test_period_batch_hashes_source_once_and_rejects_source_stat_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "periods.json"
            write_json(config_path, alive_period_payload(1900, 1901, 20, 1))
            source = root / "graph_data.pt"
            source.write_bytes(b"small fixture")
            calls = []

            def fake_build(source_data, output_root, config, period_id, **kwargs):
                calls.append((period_id, kwargs.get("_source_hash")))
                return {"period_id": period_id, "reused_existing": False}

            with mock.patch.object(artifacts, "sha256_file", return_value="one-digest") as digest, \
                    mock.patch.object(artifacts, "build_period_induced_artifact", side_effect=fake_build):
                artifacts.prepare_period_induced_artifacts(
                    source, root / "out", config_path, ("center_1900", "center_1901")
                )
            digest.assert_called_once_with(source.resolve())
            self.assertEqual(calls, [("center_1900", "one-digest"), ("center_1901", "one-digest")])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "periods.json"
            write_json(config_path, alive_period_payload(1900, 1901, 20, 1))
            source = root / "graph_data.pt"
            source.write_bytes(b"small fixture")
            calls = []

            def mutate_source(source_data, output_root, config, period_id, **kwargs):
                calls.append(period_id)
                source_data = Path(source_data)
                if len(calls) == 1:
                    source_data.write_bytes(source_data.read_bytes() + b" changed")
                return {"period_id": period_id, "reused_existing": False}

            with mock.patch.object(artifacts, "sha256_file", return_value="one-digest"), \
                    mock.patch.object(artifacts, "build_period_induced_artifact", side_effect=mutate_source):
                with self.assertRaisesRegex(ValueError, "Source graph changed during period preparation"):
                    artifacts.prepare_period_induced_artifacts(
                        source, root / "out", config_path, ("center_1900", "center_1901")
                    )
            self.assertEqual(calls, ["center_1900"])


if __name__ == "__main__":
    unittest.main()
