#!/usr/bin/env python3
"""Unit and integration tests for the FlowTwin-Guard research model."""

from __future__ import annotations

import importlib.util
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_DIR))

from flowtwin_guard.conformal import (  # noqa: E402
    ModeConformalCalibrator,
    corrected_quantile,
)
from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph  # noqa: E402
from ml_pipeline_common import load_contract, sha256_file, verify_checksums  # noqa: E402


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch  # noqa: E402

    from flowtwin_guard.data import (  # noqa: E402
        FeatureStandardizer,
        HTSTWindowDataset,
        _sample_window_ranges,
        load_target_contract,
        normalized_sensor_uncertainty,
    )
    from flowtwin_guard.model import (  # noqa: E402
        FlowTwinConfig,
        FlowTwinGuard,
        RouteGate,
        fractional_delay,
    )


class FlowTwinGraphTests(unittest.TestCase):
    def test_graph_uses_only_allowlisted_observables(self) -> None:
        graph = build_process_graph()
        contract = load_contract()
        self.assertEqual(len(graph.nodes), 22)
        self.assertEqual(len(graph.edges), 23)
        self.assertEqual(len(graph.features), 29)
        self.assertEqual(sum(graph.sensor_uncertainty_known), 6)
        self.assertFalse(set(graph.features) & set(contract["always_forbidden_features"]))
        for forbidden in (
            "plant_mode",
            "time_s",
            "fault_active",
            "fdv_actual_forward",
            "unsafe_forward_l",
            "balance_tank_volume_l",
            "inlet_temp_c",
        ):
            self.assertNotIn(forbidden, graph.features)

    def test_profile_overlay_preserves_exact_fifo_volumes(self) -> None:
        graph = build_process_graph()
        profile = {
            "nominal_flow_l_h": 20_000.0,
            "nominal_holding_time_s": 18.0,
            "sensor_to_fdv_delay_s": 1.5,
            "post_fdv_residence_time_s": 5.0,
        }
        volume = dict(zip(graph.edge_tags, graph.edge_volumes_for_profile(profile)))
        self.assertAlmostEqual(volume["L-005"], 100.0)
        self.assertAlmostEqual(volume["L-007"], 8.333333333333334)
        self.assertAlmostEqual(
            volume["L-008"] + volume["L-009"] + volume["L-010"],
            27.77777777777778,
        )
        self.assertAlmostEqual(3600.0 * volume["L-007"] / 20_000.0, 1.5)
        delay_kind = {edge.tag: edge.delay_kind for edge in graph.edges}
        self.assertEqual(delay_kind["L-011"], "exact_one_step_return")
        self.assertEqual(delay_kind["L-004"], "nominal_pid_prior")


class FlowTwinConformalTests(unittest.TestCase):
    def test_corrected_quantile_and_fail_closed_decisions(self) -> None:
        self.assertAlmostEqual(
            corrected_quantile([value / 10 for value in range(1, 10)], 0.2), 0.8
        )
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]] * 10,
            dtype=np.float64,
        )
        targets = np.zeros(40, dtype=np.int64)
        modes = [0] * 20 + [1] * 20
        energies = np.linspace(-3.0, -2.0, 40)
        calibrator = ModeConformalCalibrator(
            alpha=0.2, ood_alpha=0.1, minimum_mode_samples=5
        ).fit(
            probabilities,
            targets,
            modes,
            energies,
            splits=["validation_id"] * 40,
        )
        singleton = calibrator.predict([0.99, 0.01], 0, -3.0)
        self.assertEqual(singleton.prediction_set, (0,))
        self.assertEqual(singleton.decision, "DIAGNOSE")
        ood = calibrator.predict([0.99, 0.01], 0, 100.0)
        self.assertEqual(ood.decision, "UNKNOWN")
        state = calibrator.to_dict()
        self.assertNotIn("SAFE", state["decision_vocabulary"])
        self.assertNotIn("RELEASE", state["decision_vocabulary"])
        restored = ModeConformalCalibrator.from_dict(state)
        self.assertEqual(restored.predict([0.99, 0.01], 0, -3.0), singleton)

    def test_calibration_rejects_non_validation_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_id only"):
            ModeConformalCalibrator().fit(
                np.asarray([[0.8, 0.2], [0.7, 0.3]]),
                np.asarray([0, 0]),
                [0, 0],
                np.asarray([-2.0, -2.1]),
                splits=["train_id", "validation_id"],
            )


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch ML dependency is not installed")
class FlowTwinTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.graph = build_process_graph()

    def _controls(self, batch: int = 1, steps: int = 1) -> torch.Tensor:
        controls = torch.zeros(batch, steps, len(CONTROL_FIELDS))
        controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 20_000.0
        controls[..., CONTROL_FIELDS.index("power_good_signal")] = 1.0
        return controls

    def test_per_episode_window_sampling_spans_timeline(self) -> None:
        ranges = tuple((index, index + 8, 0) for index in range(21))
        selected = _sample_window_ranges(ranges, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0], ranges[0])
        self.assertEqual(selected[-1], ranges[-1])
        self.assertEqual(_sample_window_ranges(ranges, 0), ranges)
        event_selected = _sample_window_ranges(
            ranges, 5, priority_positions=(11, 12)
        )
        self.assertTrue(
            any(start <= 11 < end for start, end, _new_start in event_selected)
        )
        singleton = _sample_window_ranges(
            ranges, 1, priority_positions=(18,)
        )
        self.assertTrue(singleton[0][0] <= 18 < singleton[0][1])

        overlapping = ((0, 8, 0), (4, 12, 4), (8, 16, 4), (12, 20, 4))
        owning = _sample_window_ranges(
            overlapping, 1, priority_positions=(6,)
        )
        start, end, new_start = owning[0]
        self.assertTrue(start + new_start <= 6 < end)

    def test_split_firewall_prevents_rejected_episode_materialization(self) -> None:
        targets = load_target_contract(graph=self.graph)
        standardizer = FeatureStandardizer(
            tuple(self.graph.features),
            2,
            np.zeros(len(self.graph.features), dtype=np.float64),
            np.ones(len(self.graph.features), dtype=np.float64),
        )
        episodes = {
            "train-episode": {
                "episode_id": "train-episode",
                "plant_profile_id": "train-profile",
                "counterfactual_group_id": "train-group",
                "canonical_scenario": "normal",
                "counterfactual_reference": "normal",
            },
            "test-episode": {
                "episode_id": "test-episode",
                "plant_profile_id": "test-profile",
                "counterfactual_group_id": "test-group",
                "canonical_scenario": "normal",
                "counterfactual_reference": "normal",
            },
        }
        profiles = {
            "train-profile": {},
            "test-profile": {},
        }
        split_map = {
            "train-episode": "train_id",
            "test-episode": "test_id",
        }

        def fake_episode(episode_id: str, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                episode_id=episode_id,
                split=split_map[episode_id],
                meta=episodes[episode_id],
            )

        def fake_cached_episode(
            _cache: Path,
            entry: dict[str, str],
            **_kwargs: object,
        ) -> object:
            return fake_episode(entry["episode_id"])

        with tempfile.TemporaryDirectory(prefix="flowtwin-firewall-") as temporary:
            root = Path(temporary)
            source = root / "dataset"
            source.mkdir()
            fields = ["episode_id", "time_start_s", "time_s"]
            for filename in ("signals.csv", "oracle_labels.csv"):
                with (source / filename).open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(
                        [
                            {
                                "episode_id": "train-episode",
                                "time_start_s": "0",
                                "time_s": "1",
                            },
                            {
                                "episode_id": "test-episode",
                                "time_start_s": "0",
                                "time_s": "1",
                            },
                        ]
                    )
            uncached = HTSTWindowDataset(
                source,
                root / "splits",
                self.graph,
                targets,
                standardizer,
                accepted_splits=("train_id",),
                window_size=2,
                stride=2,
            )
            with patch(
                "flowtwin_guard.data._materialize_episode",
                side_effect=fake_episode,
            ) as materialize:
                pairs = list(uncached._episode_pairs(episodes, profiles, split_map))
            self.assertEqual([pair[0].episode_id for pair in pairs], ["train-episode"])
            self.assertEqual(
                [call.args[0] for call in materialize.call_args_list],
                ["train-episode"],
            )

            entries = [
                {
                    "episode_id": episode_id,
                    "counterfactual_group_id": meta["counterfactual_group_id"],
                    "reference_episode_id": episode_id,
                }
                for episode_id, meta in episodes.items()
            ]
            cached = HTSTWindowDataset(
                source,
                root / "splits",
                self.graph,
                targets,
                standardizer,
                accepted_splits=("train_id",),
                window_size=2,
                stride=2,
                cache=root / "cache",
                cache_manifest={"episodes": entries},
            )
            with patch(
                "flowtwin_guard.data._load_cached_episode",
                side_effect=fake_cached_episode,
            ) as load_cached:
                pairs = list(cached._episode_pairs(episodes, profiles, split_map))
            self.assertEqual([pair[0].episode_id for pair in pairs], ["train-episode"])
            self.assertEqual(
                [call.args[1]["episode_id"] for call in load_cached.call_args_list],
                ["train-episode"],
            )

    def test_fractional_delay_interpolates_and_propagates_gradient(self) -> None:
        history = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        history.requires_grad_(True)
        delayed, valid = fractional_delay(
            history, torch.full((1, 5), 1.5, dtype=torch.float32)
        )
        self.assertFalse(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1]))
        self.assertAlmostEqual(float(delayed[0, 4, 0].detach()), 2.5)
        delayed[0, 4, 0].backward()
        self.assertAlmostEqual(float(history.grad[0, 2, 0]), 0.5)
        self.assertAlmostEqual(float(history.grad[0, 3, 0]), 0.5)

    def test_route_gate_follows_feedback_not_command_or_oracle(self) -> None:
        gate = RouteGate(self.graph)
        controls = self._controls()
        controls[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 0.0
        values = gate(controls)[0, 0]
        by_tag = dict(zip(self.graph.edge_tags, values.tolist()))
        self.assertEqual(by_tag["L-008"], 0.0)
        self.assertEqual(by_tag["L-011"], 1.0)
        controls[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 0.25
        values = gate(controls)[0, 0]
        by_tag = dict(zip(self.graph.edge_tags, values.tolist()))
        self.assertAlmostEqual(by_tag["L-008"], 0.25)
        self.assertAlmostEqual(by_tag["L-011"], 0.75)
        controls[..., CONTROL_FIELDS.index("cip_cycle_active")] = 1.0
        values = gate(controls)[0, 0]
        by_tag = dict(zip(self.graph.edge_tags, values.tolist()))
        self.assertEqual(by_tag["L-008"], 0.0)
        self.assertEqual(by_tag["L-011"], 0.0)
        self.assertEqual(by_tag["L-201"], 1.0)
        self.assertEqual(by_tag["L-203"], 1.0)

    def test_model_is_causal_finite_and_trainable(self) -> None:
        targets = load_target_contract(graph=self.graph)
        config = FlowTwinConfig(
            feature_count=len(self.graph.features),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            class_count=len(targets.codes),
            location_count=len(targets.locations),
            mechanism_count=len(targets.mechanisms),
            hidden_dim=8,
            observer_hidden_dim=8,
            graph_layers=1,
            dropout=0.0,
        )
        model = FlowTwinGuard(
            self.graph, config, torch.zeros(len(self.graph.features))
        ).eval()
        batch, steps = 2, 17
        x = torch.randn(batch, steps, len(self.graph.features))
        controls = self._controls(batch, steps)
        controls[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 0.5
        dt_s = torch.ones(batch, steps)
        profile = {
            "nominal_flow_l_h": 20_000.0,
            "nominal_holding_time_s": 18.0,
            "sensor_to_fdv_delay_s": 1.5,
            "post_fdv_residence_time_s": 5.0,
        }
        volume = torch.tensor(self.graph.edge_volumes_for_profile(profile)).repeat(
            batch, 1
        )
        valid = torch.ones(batch, steps, dtype=torch.bool)
        output = model(x, controls, dt_s, volume, valid, include_node_embeddings=True)
        self.assertEqual(output["class_logits"].shape, (batch, steps, 20))
        self.assertEqual(output["node_embedding"].shape, (batch, steps, 22, 8))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        changed = x.clone()
        changed[:, 12:] += 1000.0
        changed_output = model(changed, controls, dt_s, volume, valid)
        torch.testing.assert_close(
            output["class_logits"][:, :12],
            changed_output["class_logits"][:, :12],
            rtol=0.0,
            atol=1e-6,
        )
        model.train()
        loss = model(x, controls, dt_s, volume, valid)["class_logits"].square().mean()
        loss.backward()
        graph_gradient = model.graph_blocks[0].advective_message.weight.grad
        self.assertIsNotNone(graph_gradient)
        self.assertTrue(torch.isfinite(graph_gradient).all())
        self.assertGreater(float(graph_gradient.abs().sum()), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch ML dependency is not installed")
class FlowTwinCLITests(unittest.TestCase):
    def test_end_to_end_cli_writes_reloadable_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="flowtwin-cli-") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            splits = root / "splits"
            cache = root / "cache"
            output = root / "output"
            benchmark_output = root / "benchmark-output"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            commands = [
                [
                    sys.executable,
                    str(MODULE_DIR / "generate_ml_dataset.py"),
                    "--output",
                    str(dataset),
                    "--dataset-version",
                    "D1-flowtwin-test",
                    "--profiles",
                    "3",
                    "--replicates",
                    "1",
                    "--duration-s",
                    "90",
                    "--dt-s",
                    "1",
                    "--seed",
                    "112233",
                    "--scenarios",
                    "normal",
                    "start_stop",
                    "cip_cycle",
                    "steam_loss",
                    "flow_surge",
                    "incomplete_cleaning",
                ],
                [
                    sys.executable,
                    str(MODULE_DIR / "split_ml_dataset.py"),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(splits),
                    "--seed",
                    "445566",
                ],
                [
                    sys.executable,
                    str(MODULE_DIR / "build_flowtwin_cache.py"),
                    "--dataset",
                    str(dataset),
                    "--splits",
                    str(splits),
                    "--output",
                    str(cache),
                ],
                [
                    sys.executable,
                    str(MODULE_DIR / "train_flowtwin_guard.py"),
                    "--dataset",
                    str(dataset),
                    "--splits",
                    str(splits),
                    "--output",
                    str(output),
                    "--cache",
                    str(cache),
                    "--window-size",
                    "16",
                    "--stride",
                    "16",
                    "--batch-size",
                    "4",
                    "--observer-epochs",
                    "1",
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "8",
                    "--observer-hidden-dim",
                    "8",
                    "--graph-layers",
                    "1",
                    "--dropout",
                    "0",
                    "--train-windows-per-episode",
                    "2",
                    "--seed",
                    "778899",
                ],
                [
                    sys.executable,
                    str(MODULE_DIR / "run_flowtwin_benchmark.py"),
                    "--dataset",
                    str(dataset),
                    "--splits",
                    str(splits),
                    "--output",
                    str(benchmark_output),
                    "--cache",
                    str(cache),
                    "--variants",
                    "flowtwin_guard",
                    "--seeds",
                    "778899",
                    "--window-size",
                    "16",
                    "--stride",
                    "16",
                    "--train-windows-per-episode",
                    "2",
                    "--batch-size",
                    "4",
                    "--observer-epochs",
                    "1",
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "8",
                    "--observer-hidden-dim",
                    "8",
                    "--layers",
                    "1",
                    "--dropout",
                    "0",
                ],
            ]
            for command in commands:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )
            expected = {
                "training_config.json",
                "training_history.json",
                "graph_contract.json",
                "standardizer.json",
                "model.pt",
                "conformal_state.json",
                "predictions.csv",
                "metrics.json",
                "run_manifest.json",
                "checksums.sha256",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertEqual(verify_checksums(output), [])
            self.assertEqual(verify_checksums(cache), [])
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertIsNotNone(manifest["cache_manifest_sha256"])
            self.assertEqual(manifest["input_contract"]["forbidden_feature_intersection"], [])
            self.assertEqual(manifest["split_policy"]["training"], ["train_id"])
            self.assertEqual(
                manifest["split_policy"]["conformal_calibration"], ["validation_id"]
            )
            conformal = json.loads((output / "conformal_state.json").read_text())
            self.assertEqual(conformal["calibration_split"], "validation_id")
            self.assertNotIn("SAFE", conformal["decision_vocabulary"])
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue({row["split"] for row in rows} <= {"test_id"})
            self.assertFalse({row["decision"] for row in rows} & {"SAFE", "RELEASE"})

            self.assertEqual(verify_checksums(benchmark_output), [])
            benchmark_manifest = json.loads(
                (benchmark_output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                benchmark_manifest["dataset_checksums_sha256"],
                sha256_file(dataset / "checksums.sha256"),
            )
            self.assertEqual(
                benchmark_manifest["split_checksums_sha256"],
                sha256_file(splits / "checksums.sha256"),
            )
            for source_name in (
                "ml_pipeline_common.py",
                "reference_plant.py",
                "sensor_calibration.py",
            ):
                with self.subTest(runtime_provenance=source_name):
                    self.assertEqual(
                        benchmark_manifest["provenance"][source_name],
                        sha256_file(MODULE_DIR / source_name),
                    )

            (splits / "checksums.sha256").unlink()
            missing_checksum_output = root / "missing-checksum-output"
            missing_checksum_command = list(commands[-1])
            output_index = missing_checksum_command.index("--output") + 1
            missing_checksum_command[output_index] = str(missing_checksum_output)
            rejected = subprocess.run(
                missing_checksum_command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("checksums.sha256 is missing or empty", rejected.stderr)
            self.assertFalse(missing_checksum_output.exists())


if __name__ == "__main__":
    unittest.main()
