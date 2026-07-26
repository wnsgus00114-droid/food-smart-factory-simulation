#!/usr/bin/env python3
"""Integration and leakage-contract tests for the HTST ML pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Sequence


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_DIR))

from audit_ml_dataset import run_audit  # noqa: E402
from generate_ml_dataset import _sample_profile  # noqa: E402
from ml_pipeline_common import load_contract, select_taxonomy  # noqa: E402
from model import HTSTConfig, HTSTSimulator, SCENARIO_NAMES  # noqa: E402
from run_ml_baselines import (  # noqa: E402
    _fit_stats,
    _rule_score,
    _scored_rows,
)


GENERATE = MODULE_DIR / "generate_ml_dataset.py"
SPLIT = MODULE_DIR / "split_ml_dataset.py"
AUDIT = MODULE_DIR / "audit_ml_dataset.py"
BASELINES = MODULE_DIR / "run_ml_baselines.py"


class MLPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._temporary = tempfile.TemporaryDirectory(prefix="htst-ml-pipeline-")
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.root = Path(cls._temporary.name)
        cls.dataset = cls.root / "dataset"
        cls.dataset_copy = cls.root / "dataset-copy"
        scenario_args = [
            "normal",
            "start_stop",
            "cip_cycle",
            "incomplete_cleaning",
            "steam_loss",
            "sensor_bias_high",
            "control_sensor_bias_high",
            "booster_pump_failure",
            "power_failure",
            "sensor_drift",
            "sensor_dropout",
            "slow_valve",
            "valve_leakage",
        ]
        base_args = [
            "--dataset-version",
            "D1-test",
            "--profiles",
            "3",
            "--replicates",
            "1",
            "--duration-s",
            "90",
            "--dt-s",
            "1",
            "--seed",
            "97531",
            "--scenarios",
            *scenario_args,
        ]
        cls._run([sys.executable, str(GENERATE), "--output", str(cls.dataset), *base_args])
        cls._run(
            [sys.executable, str(GENERATE), "--output", str(cls.dataset_copy), *base_args]
        )
        cls.splits = cls.root / "splits"
        cls.splits_copy = cls.root / "splits-copy"
        cls._run(
            [
                sys.executable,
                str(SPLIT),
                "--dataset",
                str(cls.dataset),
                "--output",
                str(cls.splits),
                "--seed",
                "86420",
            ]
        )
        cls._run(
            [
                sys.executable,
                str(SPLIT),
                "--dataset",
                str(cls.dataset_copy),
                "--output",
                str(cls.splits_copy),
                "--seed",
                "86420",
            ]
        )

    @staticmethod
    def _run_allow_failure(
        command: Sequence[str], timeout_s: float = 60.0
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

    @classmethod
    def _run(cls, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        process = cls._run_allow_failure(command)
        if process.returncode:
            raise AssertionError(
                f"command failed ({process.returncode}): {' '.join(command)}\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        return process

    @staticmethod
    def _rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _header(path: Path) -> list[str]:
        with path.open(newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle))

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"Expected JSON object in {path}")
        return value

    def test_v2_contract_canonicalizes_alias_and_covers_every_behavior_once(self) -> None:
        contract = load_contract()
        taxonomy = {item["code"]: item for item in contract["taxonomy"]}
        self.assertEqual(contract["contract_version"], "2.2.0")
        self.assertEqual(contract["model_version"], "2.2.0")
        self.assertEqual(len(taxonomy), 20)
        self.assertTrue(all(item["implemented"] for item in taxonomy.values()))
        self.assertEqual(taxonomy["M01"]["canonical_scenario"], "start_stop")
        self.assertEqual(taxonomy["F12"]["canonical_scenario"], "incomplete_cleaning")
        self.assertEqual(taxonomy["F13"]["canonical_scenario"], "power_failure")
        self.assertEqual(taxonomy["F14"]["canonical_scenario"], "sensor_drift")
        self.assertEqual(taxonomy["F15"]["canonical_scenario"], "sensor_dropout")
        self.assertEqual(taxonomy["F16"]["canonical_scenario"], "slow_valve")
        self.assertEqual(taxonomy["F17"]["canonical_scenario"], "valve_leakage")
        selected = select_taxonomy(
            contract, ["sensor_bias_high", "control_sensor_bias_high", "F03"]
        )
        self.assertEqual([item["code"] for item in selected], ["F03"])
        default = select_taxonomy(contract, None)
        self.assertEqual(len(default), 20)
        contracted = {
            item["simulator_scenarios"][0]
            for item in default
        }
        self.assertEqual(contracted, set(SCENARIO_NAMES) - {"sensor_bias_high"})
        for feature_set in contract["feature_sets"].values():
            self.assertNotIn("alarm_count", feature_set)
            self.assertNotIn("observable_alarm_count", feature_set)
            self.assertNotIn("alarm_unsafe_forward", feature_set)
            self.assertNotIn("differential_pressure_bar", feature_set)
            self.assertNotIn("fdv_actual_forward", feature_set)
            self.assertNotIn("sensor_available", feature_set)
        self.assertIn("fdv_actual_forward", contract["always_forbidden_features"])
        self.assertIn(
            "measured_differential_pressure_bar", contract["feature_sets"]["S1-pressure"]
        )
        self.assertIn("fdv_position_feedback", contract["feature_sets"]["S0-core"])
        self.assertIn(
            "estimated_fastest_residence_time_s",
            contract["feature_sets"]["S2-derived"],
        )
        expected_v2_profile_ranges = {
            "balance_tank_capacity_l",
            "balance_tank_initial_volume_l",
            "balance_tank_heat_loss_tau_s",
            "stationary_cooling_tau_s",
            "cip_caustic_removal_rate_s",
            "cip_acid_removal_rate_s",
            "cip_reference_velocity_m_s",
            "cip_line_hold_up_l",
        }
        self.assertTrue(
            expected_v2_profile_ranges <= set(contract["profile_parameter_ranges"])
        )
        for profile_index in range(20):
            _, _, sampled = _sample_profile(contract, 24680, profile_index)
            self.assertLessEqual(
                sampled["balance_tank_initial_volume_l"],
                sampled["balance_tank_capacity_l"],
            )
            HTSTConfig(**sampled, duration_s=90.0, dt_s=1.0)

    def test_generator_is_deterministic_and_separates_signals_from_oracles(self) -> None:
        for filename in (
            "profiles.csv",
            "episodes.csv",
            "signals.csv",
            "oracle_labels.csv",
            "dataset_schema.json",
            "dataset_manifest.json",
            "checksums.sha256",
        ):
            with self.subTest(filename=filename):
                self.assertEqual(
                    (self.dataset / filename).read_bytes(),
                    (self.dataset_copy / filename).read_bytes(),
                )
        signal_header = set(self._header(self.dataset / "signals.csv"))
        label_header = set(self._header(self.dataset / "oracle_labels.csv"))
        manifest = self._json(self.dataset / "dataset_manifest.json")
        schema = self._json(self.dataset / "dataset_schema.json")
        self.assertEqual(manifest["model_version"], "2.2.0")
        self.assertEqual(manifest["contract_version"], "2.2.0")
        self.assertEqual(manifest["generator_version"], "2.2.0")
        self.assertIn("process_components.py", manifest["provenance"])
        self.assertEqual(schema["schema_version"], "2.2.0")
        for table in (
            "profiles.csv",
            "episodes.csv",
            "signals.csv",
            "oracle_labels.csv",
        ):
            table_schema = schema["tables"][table]
            self.assertTrue(table_schema["fields"])
            self.assertTrue(
                all(
                    field["type"] in {"boolean", "integer", "number", "string"}
                    and isinstance(field["nullable"], bool)
                    for field in table_schema["fields"]
                )
            )
        for forbidden in (
            "alarm_count",
            "observable_alarm_count",
            "alarm_unsafe_forward",
            "differential_pressure_bar",
            "fdv_actual_forward",
            "power_available",
            "sensor_available",
            "valve_travel_time_factor",
            "valve_leakage_fraction",
            "cleaning_effectiveness_factor",
        ):
            self.assertNotIn(forbidden, signal_header)
            self.assertIn(forbidden, label_header)
        self.assertIn("measured_differential_pressure_bar", signal_header)
        self.assertIn("fdv_position_feedback", signal_header)
        self.assertIn("fdv_position_error", signal_header)
        self.assertNotIn("fdv_actual_forward", signal_header)
        self.assertIn("fdv_actual_forward", label_header)
        self.assertIn("differential_pressure_bar", label_header)
        self.assertIn("power_available", label_header)
        for observable in (
            "cip_cycle_active",
            "cip_chemical_concentration_pct",
            "cip_conductivity_proxy_ms_cm",
            "cip_ph_proxy",
            "cip_release_permissive",
            "post_cip_conductivity_proxy_ms_cm",
            "post_cip_ph_proxy",
            "restart_product_interface_signal_fraction",
            "power_good_signal",
            "temperature_sensor_quality_ok",
        ):
            self.assertIn(observable, signal_header)
            self.assertNotIn(observable, label_header)
        self.assertIn("physical_effect_time_s", label_header)
        self.assertIn("future_safety_event_30s", label_header)
        model_row = HTSTSimulator(
            HTSTConfig(duration_s=1.0, dt_s=1.0)
        ).run("normal")[0]
        observable_component_fields = {
            "cip_cycle_active",
            "cip_chemical_concentration_pct",
            "cip_conductivity_proxy_ms_cm",
            "cip_ph_proxy",
            "cip_release_permissive",
            "post_cip_conductivity_proxy_ms_cm",
            "post_cip_ph_proxy",
            "restart_product_interface_signal_fraction",
        }
        component_truth = {
            field
            for field in model_row
            if (
                field.startswith(("balance_tank_", "shadow_", "cip_"))
                and field not in observable_component_fields
            )
            or field
            in {
                "fresh_feed_l",
                "return_to_balance_tank_l",
                "return_line_inventory_l",
                "inlet_mean_pass_count",
                "inlet_recycle_risk_fraction",
                "inlet_chemical_fraction",
                "diverted_mean_pass_count",
                "diverted_recycle_risk_fraction",
            }
        }
        contract = load_contract()
        self.assertTrue(component_truth)
        self.assertTrue(component_truth <= set(contract["always_forbidden_features"]))
        self.assertTrue(component_truth <= label_header)
        self.assertFalse(component_truth & signal_header)
        self.assertEqual(
            signal_header & label_header,
            {"episode_id", "time_start_s", "time_s"},
        )

        episodes = self._rows(self.dataset / "episodes.csv")
        self.assertEqual(len(episodes), 3 * 12)
        self.assertNotIn("sensor_bias_high", {row["simulator_scenario"] for row in episodes})
        f03 = [row for row in episodes if row["canonical_code"] == "F03"]
        self.assertEqual(len(f03), 3)
        self.assertEqual({row["canonical_scenario"] for row in f03}, {"control_sensor_bias_high"})
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in episodes:
            groups[row["counterfactual_group_id"]].append(row)
        self.assertEqual(len(groups), 3)
        for rows in groups.values():
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({row["noise_seed"] for row in rows}), 1)
        fault_rows = [row for row in episodes if row["is_fault"] == "1"]
        self.assertGreater(len({row["fault_injection_time_s"] for row in fault_rows}), 1)
        self.assertGreater(len({row["fault_severity"] for row in fault_rows}), 1)
        for row in episodes:
            if row["canonical_code"] == "N00":
                self.assertFalse(row["behavior_effect_time_s"])
                continue
            self.assertLessEqual(
                float(row["behavior_injection_time_s"]),
                float(row["physical_effect_time_s"]),
            )
            self.assertLessEqual(
                float(row["behavior_injection_time_s"]),
                float(row["behavior_effect_time_s"]),
            )
        f12 = [row for row in episodes if row["canonical_code"] == "F12"]
        self.assertEqual({row["counterfactual_reference"] for row in f12}, {"cip_cycle"})
        self.assertTrue(
            all(float(row["fault_injection_time_s"]) >= 60.0 for row in f12)
        )
        m01 = [row for row in episodes if row["canonical_code"] == "M01"]
        self.assertEqual({row["is_fault"] for row in m01}, {"0"})
        self.assertTrue(all(row["behavior_effect_time_s"] for row in m01))
        labels = self._rows(self.dataset / "oracle_labels.csv")
        m01_ids = {row["episode_id"] for row in m01}
        m01_labels = [row for row in labels if row["episode_id"] in m01_ids]
        self.assertTrue(any(row["behavior_active"] == "1" for row in m01_labels))
        self.assertTrue(all(row["fault_active"] == "0" for row in m01_labels))
        signals = self._rows(self.dataset / "signals.csv")
        signal_mode = {
            (row["episode_id"], row["time_s"]): row["plant_mode"] for row in signals
        }
        by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in m01_labels:
            by_episode[row["episode_id"]].append(row)
        for episode_rows in by_episode.values():
            previous_mode: str | None = None
            for row in episode_rows:
                mode = signal_mode[(row["episode_id"], row["time_s"])]
                expected = previous_mode is not None and mode != previous_mode
                self.assertEqual(row["mode_transition_event"] == "1", expected)
                previous_mode = mode

        episode_meta = {row["episode_id"]: row for row in episodes}
        future_fields = [field for field in labels[0] if field.startswith("future_")]
        saw_censored = False
        saw_observed = False
        for row in labels:
            duration_s = float(episode_meta[row["episode_id"]]["duration_s"])
            time_s = float(row["time_s"])
            for field in future_fields:
                horizon_s = float(field.rsplit("_", 1)[1][:-1])
                if time_s + horizon_s > duration_s + 1e-12:
                    self.assertEqual(row[field], "")
                    saw_censored = True
                else:
                    self.assertIn(row[field], {"0", "1"})
                    saw_observed = True
            expected_cooling = int(
                float(row["product_boundary_l"]) > 1e-12
                and float(row["product_temp_c"])
                > float(load_contract()["baseline_thresholds"]["maximum_product_temperature_c"])
            )
            self.assertEqual(int(row["cooling_excursion"]), expected_cooling)
        self.assertTrue(saw_censored and saw_observed)
        f12_by_id = {row["episode_id"]: row for row in f12}
        for row in labels:
            meta = f12_by_id.get(row["episode_id"])
            if meta is not None and row["fault_active"] == "1":
                self.assertGreaterEqual(
                    float(row["time_start_s"]),
                    float(meta["fault_injection_time_s"]),
                )

    def test_profile_split_is_deterministic_and_keeps_related_groups_together(self) -> None:
        for filename in ("episode_splits.csv", "split_manifest.json", "checksums.sha256"):
            self.assertEqual(
                (self.splits / filename).read_bytes(),
                (self.splits_copy / filename).read_bytes(),
            )
        rows = self._rows(self.splits / "episode_splits.csv")
        self.assertEqual({row["split"] for row in rows}, {"train_id", "validation_id", "test_id"})
        for field in (
            "plant_profile_id",
            "counterfactual_group_id",
            "profile_config_hash",
            "noise_seed",
            "sampling_seed",
        ):
            owners: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                owners[row[field]].add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()), field)

    def test_auditor_passes_clean_dataset_and_rejects_privileged_column(self) -> None:
        report_path = self.root / "audit.json"
        self._run(
            [
                sys.executable,
                str(AUDIT),
                "--dataset",
                str(self.dataset),
                "--splits",
                str(self.splits),
                "--report",
                str(report_path),
            ]
        )
        report = self._json(report_path)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["errors"])
        self.assertIn("audit_ml_dataset.py", report["provenance"])
        checksum_path = report_path.with_name(report_path.name + ".sha256")
        digest, separator, name = checksum_path.read_text(encoding="utf-8").strip().partition("  ")
        self.assertEqual(separator, "  ")
        self.assertEqual(name, report_path.name)
        self.assertEqual(digest, hashlib.sha256(report_path.read_bytes()).hexdigest())

        corrupted = self.root / "corrupted"
        shutil.copytree(self.dataset, corrupted)
        signal_path = corrupted / "signals.csv"
        lines = signal_path.read_text(encoding="utf-8").splitlines()
        lines[0] += ",alarm_count"
        lines[1:] = [line + ",0" for line in lines[1:]]
        signal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        bad_report = run_audit(
            argparse.Namespace(
                dataset=corrupted,
                splits=self.splits,
                windows=None,
                report=None,
            )
        )
        self.assertEqual(bad_report["status"], "failed")
        self.assertTrue(
            any("explicit_privileged_fields_absent" in error for error in bad_report["errors"])
        )

    def test_auditor_rejects_cross_split_overlapping_windows(self) -> None:
        split_rows = self._rows(self.splits / "episode_splits.csv")
        episode = split_rows[0]
        wrong_split = "test_id" if episode["split"] != "test_id" else "train_id"
        windows = self.root / "bad-windows.csv"
        with windows.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "window_id",
                    "episode_id",
                    "window_start_s",
                    "window_end_s",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "window_id": "w1",
                    "episode_id": episode["episode_id"],
                    "window_start_s": 10,
                    "window_end_s": 30,
                    "split": episode["split"],
                }
            )
            writer.writerow(
                {
                    "window_id": "w2",
                    "episode_id": episode["episode_id"],
                    "window_start_s": 20,
                    "window_end_s": 40,
                    "split": wrong_split,
                }
            )
        report = run_audit(
            argparse.Namespace(
                dataset=self.dataset,
                splits=self.splits,
                windows=windows,
                report=None,
            )
        )
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("window_overlap_isolation" in error for error in report["errors"])
        )

    def test_baseline_scores_do_not_depend_on_plant_mode(self) -> None:
        mutated = self.root / "plant-mode-mutated"
        shutil.copytree(self.dataset, mutated)
        signal_path = mutated / "signals.csv"
        rows = self._rows(signal_path)
        with signal_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for index, row in enumerate(rows):
                row["plant_mode"] = f"UNTRUSTED_CONTEXT_{index % 7}"
                writer.writerow(row)

        episode_meta = {
            row["episode_id"]: row
            for row in self._rows(self.dataset / "episodes.csv")
        }
        episode_split = {
            row["episode_id"]: row["split"]
            for row in self._rows(self.splits / "episode_splits.csv")
        }
        original_stats = _fit_stats(self.dataset, episode_meta, episode_split)
        mutated_stats = _fit_stats(mutated, episode_meta, episode_split)
        self.assertEqual(set(original_stats), {"__GLOBAL__", "cip", "production"})
        for regime in original_stats:
            for feature in original_stats[regime]:
                left = original_stats[regime][feature]
                right = mutated_stats[regime][feature]
                self.assertEqual(
                    (left.count, left.mean, left.m2),
                    (right.count, right.mean, right.m2),
                )

        thresholds = load_contract()["baseline_thresholds"]
        original_scores = [
            scores
            for _, _, scores in _scored_rows(
                self.dataset, original_stats, thresholds, 0.2, 0.5
            )
        ]
        mutated_scores = [
            scores
            for _, _, scores in _scored_rows(
                mutated, mutated_stats, thresholds, 0.2, 0.5
            )
        ]
        self.assertEqual(original_scores, mutated_scores)

    def test_rule_uses_observable_cip_gate_and_chemistry_response(self) -> None:
        signal = self._rows(self.dataset / "signals.csv")[0]
        thresholds = load_contract()["baseline_thresholds"]
        signal.update(
            {
                "cip_cycle_active": "1",
                "cip_chemical_concentration_pct": "0",
                "cip_conductivity_proxy_ms_cm": "0.2",
                "cip_ph_proxy": "7.0",
                "power_good_signal": "1",
                "temperature_sensor_quality_ok": "1",
                # These are production violations and must be gated during CIP.
                "safety_temp_sensor_c": "-100",
                "estimated_fastest_residence_time_s": "0",
                "measured_flow_l_h": "1000000000",
                "measured_differential_pressure_bar": "-100",
                "sensor_disagreement_c": "100",
                "leak_detector_signal_fraction": "1",
                "fdv_position_feedback": "1",
                "product_temp_sensor_c": "100",
            }
        )
        self.assertEqual(_rule_score(signal, thresholds, 1000.0), 0.0)

        signal["cip_chemical_concentration_pct"] = "2.0"
        grace_s = float(thresholds["cip_response_grace_s"])
        self.assertEqual(_rule_score(signal, thresholds, grace_s - 1.0), 0.0)
        self.assertEqual(_rule_score(signal, thresholds, grace_s), 1.0)

        signal["cip_conductivity_proxy_ms_cm"] = "1.5"
        signal["cip_ph_proxy"] = "11.5"
        self.assertEqual(_rule_score(signal, thresholds, grace_s), 0.0)
        signal["power_good_signal"] = "0"
        self.assertEqual(_rule_score(signal, thresholds, grace_s), 1.0)

    def test_baselines_emit_event_metrics_without_privileged_inputs(self) -> None:
        output = self.root / "baselines"
        output_copy = self.root / "baselines-copy"
        for target in (output, output_copy):
            self._run(
                [
                    sys.executable,
                    str(BASELINES),
                    "--dataset",
                    str(self.dataset),
                    "--splits",
                    str(self.splits),
                    "--output",
                    str(target),
                    "--reservoir-size",
                    "1000",
                ]
            )
        for filename in (
            "baseline_scores.csv",
            "episode_metrics.csv",
            "baseline_results.json",
            "baseline_manifest.json",
            "checksums.sha256",
        ):
            self.assertEqual(
                (output / filename).read_bytes(), (output_copy / filename).read_bytes()
            )
        results = self._json(output / "baseline_results.json")
        self.assertEqual(set(results["methods"]), {"rule", "ewma", "cusum"})
        self.assertFalse(
            {
                "alarm_count",
                "observable_alarm_count",
                "alarm_unsafe_forward",
                "differential_pressure_bar",
                "fdv_actual_forward",
                "plant_mode",
                "power_available",
                "sensor_available",
            }
            & set(results["inputs"])
        )
        self.assertIn("measured_differential_pressure_bar", results["inputs"])
        self.assertIn("fdv_position_feedback", results["inputs"])
        for observable in (
            "cip_cycle_active",
            "cip_chemical_concentration_pct",
            "cip_conductivity_proxy_ms_cm",
            "cip_ph_proxy",
            "power_good_signal",
            "temperature_sensor_quality_ok",
        ):
            self.assertIn(observable, results["inputs"])
        self.assertEqual(results["baseline_version"], "2.2.0")
        self.assertEqual(results["normal_behavior_codes"], ["M01", "M02", "N00"])
        self.assertEqual(
            results["calibration"]["population"]["normal_behavior_codes"],
            ["M01", "M02", "N00"],
        )
        self.assertEqual(
            results["operational_gating"],
            {
                "field": "cip_cycle_active",
                "plant_mode_used_for_scoring": False,
                "plant_mode_used_for_standardization": False,
                "plant_mode_used_for_state_reset": False,
            },
        )
        self.assertEqual(results["counts"]["score_rows"], 3 * 12 * 90)
        for split in ("train_id", "validation_id", "test_id"):
            self.assertEqual(
                set(results["metrics_by_split"][split]), {"rule", "ewma", "cusum"}
            )
            for method in results["metrics_by_split"][split].values():
                self.assertIn("event_recall", method)
                self.assertIn("detection_latency_p90_s", method)
                self.assertIn("false_alarm_edges_per_production_hour", method)
                self.assertIn("false_alarm_edges_per_cip_hour", method)
                self.assertIn("false_alarm_cip_fraction", method)
                self.assertIn("unsafe_forward_l_before_detection", method)
                self.assertTrue(method["evaluation_population"].startswith("all rows"))

        episode_metrics = self._rows(output / "episode_metrics.csv")
        normal_cip = [
            row for row in episode_metrics if row["canonical_code"] == "M02"
        ]
        self.assertTrue(normal_cip)
        self.assertTrue(
            all(float(row["negative_cip_seconds"]) > 0.0 for row in normal_cip)
        )
        self.assertTrue(
            all(float(row["negative_production_seconds"]) == 0.0 for row in normal_cip)
        )
        self.assertTrue(
            all(row["false_alarm_cip_seconds"] != "" for row in normal_cip)
        )
        normal_production = [
            row for row in episode_metrics if row["canonical_code"] == "N00"
        ]
        self.assertTrue(normal_production)
        self.assertTrue(
            all(
                float(row["negative_production_seconds"]) > 0.0
                for row in normal_production
            )
        )

    def test_incomplete_cleaning_rejects_episode_without_active_cip_phase(self) -> None:
        process = self._run_allow_failure(
            [
                sys.executable,
                str(GENERATE),
                "--output",
                str(self.root / "must-not-exist"),
                "--profiles",
                "1",
                "--replicates",
                "1",
                "--duration-s",
                "30",
                "--scenarios",
                "F12",
            ]
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("requires duration-s to extend", process.stderr)


if __name__ == "__main__":
    unittest.main()
