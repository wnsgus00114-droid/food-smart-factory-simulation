#!/usr/bin/env python3
"""Deterministic integration and censor-contract tests for isolated D3-RUL."""

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

from audit_d3_rul_dataset import run_audit  # noqa: E402
from d3_rul_common import (  # noqa: E402
    evaluate_eol_trace,
    load_d3_contract,
    model_version_is_compatible,
    sha256_json,
    write_checksums,
)
from model import MODEL_VERSION  # noqa: E402


GENERATE = MODULE_DIR / "generate_d3_rul_dataset.py"
SPLIT = MODULE_DIR / "split_d3_rul_dataset.py"
AUDIT = MODULE_DIR / "audit_d3_rul_dataset.py"
BASELINES = MODULE_DIR / "run_d3_rul_baselines.py"


class D3RULPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._temporary = tempfile.TemporaryDirectory(prefix="htst-d3-rul-")
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.root = Path(cls._temporary.name)
        cls.dataset = cls.root / "dataset"
        cls.dataset_copy = cls.root / "dataset-copy"
        generation = [
            "--dataset-version",
            "D3-RUL-test",
            "--profiles",
            "3",
            "--trajectories-per-profile",
            "2",
            "--dt-s",
            "1",
            "--sample-interval-s",
            "10",
            "--maximum-sim-duration-s",
            "1200",
            "--accelerations",
            "100",
            "--censor-horizons-equivalent-h",
            "8",
            "28",
            "--seed",
            "123",
        ]
        cls._run(
            [sys.executable, str(GENERATE), "--output", str(cls.dataset), *generation]
        )
        cls._run(
            [
                sys.executable,
                str(GENERATE),
                "--output",
                str(cls.dataset_copy),
                *generation,
            ]
        )
        cls.splits = cls.root / "splits"
        cls.splits_copy = cls.root / "splits-copy"
        for dataset, output in (
            (cls.dataset, cls.splits),
            (cls.dataset_copy, cls.splits_copy),
        ):
            cls._run(
                [
                    sys.executable,
                    str(SPLIT),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--seed",
                    "456",
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
    def _json(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"Expected JSON object: {path}")
        return value

    def test_generator_is_byte_deterministic_and_runtime_model_compatible(self) -> None:
        for filename in (
            "profiles.csv",
            "trajectories.csv",
            "signals.csv",
            "oracle_labels.csv",
            "eol_evidence.csv",
            "dataset_schema.json",
            "dataset_manifest.json",
            "checksums.sha256",
        ):
            with self.subTest(filename=filename):
                self.assertEqual(
                    (self.dataset / filename).read_bytes(),
                    (self.dataset_copy / filename).read_bytes(),
                )
        contract = load_d3_contract()
        manifest = self._json(self.dataset / "dataset_manifest.json")
        self.assertEqual(manifest["dataset_family"], "D3-RUL")
        self.assertEqual(manifest["model_version"], MODEL_VERSION)
        self.assertTrue(model_version_is_compatible(MODEL_VERSION, contract))
        self.assertGreater(manifest["counts"]["events"], 0)
        self.assertGreater(manifest["counts"]["right_censored"], 0)
        schema = self._json(self.dataset / "dataset_schema.json")
        signal_roles = {
            field["name"]: field["role"]
            for field in schema["tables"]["signals.csv"]["fields"]
        }
        self.assertTrue(
            all(
                signal_roles[name] == "context"
                for name in contract["signal_context_fields"]
            )
        )
        self.assertTrue(
            all(
                signal_roles[name] == "observable_feature"
                for name in manifest["model_feature_fields"]
            )
        )

    def test_ood_profiles_are_sampled_outside_id_support(self) -> None:
        dataset = self.root / "ood-dataset"
        splits = self.root / "ood-splits"
        self._run(
            [
                sys.executable,
                str(GENERATE),
                "--output",
                str(dataset),
                "--dataset-version",
                "D3-RUL-ood-test",
                "--profiles",
                "4",
                "--trajectories-per-profile",
                "1",
                "--ood-profiles",
                "1",
                "--dt-s",
                "1",
                "--sample-interval-s",
                "10",
                "--maximum-sim-duration-s",
                "300",
                "--accelerations",
                "100",
                "--censor-horizons-equivalent-h",
                "8",
                "--seed",
                "321",
            ]
        )
        self._run(
            [
                sys.executable,
                str(SPLIT),
                "--dataset",
                str(dataset),
                "--output",
                str(splits),
                "--seed",
                "654",
            ]
        )
        contract = load_d3_contract()
        profiles = self._rows(dataset / "profiles.csv")
        ood = [row for row in profiles if row["domain"] == "OOD"]
        self.assertEqual(len(ood), 1)
        for name, bounds in contract["ood_profile_parameter_ranges"].items():
            value = float(ood[0][name])
            self.assertGreaterEqual(value, float(bounds[0]))
            self.assertLessEqual(value, float(bounds[1]))
            id_low, id_high = (
                float(item) for item in contract["profile_parameter_ranges"][name]
            )
            self.assertTrue(value < id_low or value > id_high)
        report = run_audit(
            argparse.Namespace(dataset=dataset, splits=splits, report=None)
        )
        self.assertEqual(report["status"], "passed")

    def test_signals_exclude_oracles_and_censored_rul_is_null(self) -> None:
        contract = load_d3_contract()
        signals = self._rows(self.dataset / "signals.csv")
        labels = self._rows(self.dataset / "oracle_labels.csv")
        trajectories = {
            row["trajectory_id"]: row
            for row in self._rows(self.dataset / "trajectories.csv")
        }
        signal_header = set(signals[0])
        label_header = set(labels[0])
        self.assertFalse(
            signal_header & set(contract["always_forbidden_signal_fields"])
        )
        self.assertEqual(
            (signal_header & label_header),
            {"trajectory_id", "time_start_sim_s", "time_sim_s"},
        )
        self.assertIn("heater_power_sensor_kw", signal_header)
        self.assertIn("operating_time_meter_h", signal_header)
        self.assertNotIn("degradation_acceleration_factor", signal_header)
        for signal in signals:
            self.assertAlmostEqual(
                float(signal["operating_time_meter_h"]),
                float(signal["time_sim_s"]) / 3600.0,
            )
        previous_fouling: dict[str, float] = {}
        saw_event = saw_censor = False
        for label in labels:
            trajectory = trajectories[label["trajectory_id"]]
            fouling = float(label["fouling_index"])
            self.assertAlmostEqual(float(label["health_index"]), 1.0 - fouling)
            self.assertGreaterEqual(
                fouling + 1.0e-12,
                previous_fouling.get(label["trajectory_id"], fouling),
            )
            previous_fouling[label["trajectory_id"]] = fouling
            if trajectory["event_observed"] == "1":
                saw_event = True
                self.assertTrue(label["rul_equivalent_s"])
                self.assertFalse(label["rul_lower_bound_equivalent_s"])
            else:
                saw_censor = True
                self.assertFalse(label["rul_equivalent_s"])
                self.assertTrue(label["rul_lower_bound_equivalent_s"])
                self.assertNotEqual(label["rul_lower_bound_equivalent_s"], "")
        self.assertTrue(saw_event and saw_censor)

    def test_profile_split_is_deterministic_and_isolates_life_families(self) -> None:
        for filename in (
            "trajectory_splits.csv",
            "split_manifest.json",
            "checksums.sha256",
        ):
            self.assertEqual(
                (self.splits / filename).read_bytes(),
                (self.splits_copy / filename).read_bytes(),
            )
        rows = self._rows(self.splits / "trajectory_splits.csv")
        self.assertEqual(
            {row["split"] for row in rows},
            {"train_id", "validation_id", "test_id"},
        )
        for field in (
            "plant_profile_id",
            "life_family_id",
            "profile_config_hash",
            "noise_seed",
            "degradation_seed",
            "censor_seed",
        ):
            owners: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                owners[row[field]].add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()), field)

    def test_auditor_passes_and_rejects_zero_filled_censored_rul(self) -> None:
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
        checksum = report_path.with_name(report_path.name + ".sha256")
        digest, separator, filename = checksum.read_text(encoding="utf-8").strip().partition("  ")
        self.assertEqual(separator, "  ")
        self.assertEqual(filename, report_path.name)
        self.assertEqual(digest, hashlib.sha256(report_path.read_bytes()).hexdigest())

        corrupted = self.root / "corrupted-censor"
        shutil.copytree(self.dataset, corrupted)
        label_path = corrupted / "oracle_labels.csv"
        rows = self._rows(label_path)
        censored = next(row for row in rows if row["event_observed"] == "0")
        censored["rul_equivalent_s"] = "0"
        with label_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checksum_names = [
            line.partition("  ")[2]
            for line in (corrupted / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        ]
        write_checksums(corrupted, checksum_names)
        bad = run_audit(
            argparse.Namespace(dataset=corrupted, splits=self.splits, report=None)
        )
        self.assertEqual(bad["status"], "failed")
        self.assertTrue(
            any("right_censoring_and_rul_semantics" in error for error in bad["errors"])
        )

    def test_auditor_rejects_profile_cross_split_leakage(self) -> None:
        corrupted = self.root / "corrupted-splits"
        shutil.copytree(self.splits, corrupted)
        split_path = corrupted / "trajectory_splits.csv"
        rows = self._rows(split_path)
        first_profile = rows[0]["plant_profile_id"]
        related = [row for row in rows if row["plant_profile_id"] == first_profile]
        self.assertGreaterEqual(len(related), 2)
        related[1]["split"] = (
            "test_id" if related[0]["split"] != "test_id" else "train_id"
        )
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checksum_names = [
            line.partition("  ")[2]
            for line in (corrupted / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        ]
        write_checksums(corrupted, checksum_names)
        bad = run_audit(
            argparse.Namespace(dataset=self.dataset, splits=corrupted, report=None)
        )
        self.assertEqual(bad["status"], "failed")
        self.assertTrue(
            any("split_isolation_plant_profile_id" in error for error in bad["errors"])
        )

    def test_auditor_reconstructs_profile_values_from_seed(self) -> None:
        corrupted = self.root / "corrupted-profile"
        shutil.copytree(self.dataset, corrupted)
        profile_path = corrupted / "profiles.csv"
        rows = self._rows(profile_path)
        rows[0]["nominal_flow_l_h"] = str(
            float(rows[0]["nominal_flow_l_h"]) + 1.0
        )
        contract = load_d3_contract()
        values = {
            name: float(rows[0][name])
            for name in sorted(contract["profile_parameter_ranges"])
        }
        rows[0]["profile_config_hash"] = sha256_json(values)
        with profile_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checksum_names = [
            line.partition("  ")[2]
            for line in (corrupted / "checksums.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        write_checksums(corrupted, checksum_names)
        bad = run_audit(
            argparse.Namespace(dataset=corrupted, splits=self.splits, report=None)
        )
        self.assertEqual(bad["status"], "failed")
        self.assertTrue(
            any("profile_sampling_and_hashes" in error for error in bad["errors"])
        )

    def test_auditor_independently_recomputes_eol_evidence(self) -> None:
        corrupted = self.root / "corrupted-eol-evidence"
        shutil.copytree(self.dataset, corrupted)
        evidence_path = corrupted / "eol_evidence.csv"
        rows = self._rows(evidence_path)
        self.assertTrue(rows)
        # Moving the claimed onset earlier still satisfies the evidence file's
        # own inequality, but it must not match a replay of the emitted trace.
        rows[0]["persistence_start_sim_s"] = str(
            float(rows[0]["persistence_start_sim_s"]) - 1.0
        )
        with evidence_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        checksum_names = [
            line.partition("  ")[2]
            for line in (corrupted / "checksums.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        write_checksums(corrupted, checksum_names)
        bad = run_audit(
            argparse.Namespace(dataset=corrupted, splits=self.splits, report=None)
        )
        self.assertEqual(bad["status"], "failed")
        self.assertTrue(
            any("eol_persistence_and_first_cause" in error for error in bad["errors"])
        )

    def test_eol_persistence_for_each_composite_condition(self) -> None:
        eol = dict(load_d3_contract()["eol"])
        eol.update(
            {
                "minimum_monitor_time_sim_s": 0.0,
                "forward_arm_persistence_sim_s": 2.0,
                "steam_persistence_sim_s": 3.0,
                "energy_persistence_sim_s": 3.0,
                "divert_persistence_sim_s": 2.0,
            }
        )

        def trace(cause: str) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for second in range(1, 9):
                active = second >= 4
                rows.append(
                    {
                        "time_start_sim_s": second - 1,
                        "time_sim_s": second,
                        "step_dt_sim_s": 1,
                        "fouling_index": 0.81 if cause == "FOULING_LIMIT" and active else 0.1,
                        "steam_valve": 0.95 if cause == "STEAM_SATURATION" and active else 0.5,
                        "actual_energy_intensity_ratio": 1.3 if cause == "ENERGY_INTENSITY" and active else 1.0,
                        "fdv_actual_forward": 0 if cause == "LOSS_OF_FORWARD" and active else 1,
                    }
                )
            return rows

        expected_trigger = {
            "FOULING_LIMIT": 4.0,
            "STEAM_SATURATION": 6.0,
            "ENERGY_INTENSITY": 6.0,
            "LOSS_OF_FORWARD": 5.0,
        }
        for cause, trigger_time in expected_trigger.items():
            with self.subTest(cause=cause):
                _, evidence, outcome = evaluate_eol_trace(trace(cause), 10.0, eol)
                self.assertIsNotNone(outcome)
                self.assertEqual(outcome["eol_cause"], cause)
                self.assertEqual(evidence[cause]["trigger_time_sim_s"], trigger_time)
                self.assertEqual(
                    evidence[cause]["trigger_time_equivalent_s"],
                    trigger_time * 10.0,
                )

    def test_fouling_limit_cannot_trigger_before_forward_arm(self) -> None:
        eol = dict(load_d3_contract()["eol"])
        eol.update(
            {
                "minimum_monitor_time_sim_s": 0.0,
                "forward_arm_persistence_sim_s": 10.0,
            }
        )
        rows = [
            {
                "time_start_sim_s": second - 1,
                "time_sim_s": second,
                "step_dt_sim_s": 1,
                "fouling_index": 0.9,
                "steam_valve": 0.5,
                "actual_energy_intensity_ratio": 1.0,
                "fdv_actual_forward": 1,
            }
            for second in range(1, 6)
        ]
        trace, evidence, outcome = evaluate_eol_trace(rows, 10.0, eol)
        self.assertTrue(all(not item["eol_monitor_armed"] for item in trace))
        self.assertFalse(evidence)
        self.assertIsNone(outcome)

    def test_baselines_are_deterministic_and_never_impute_censored_rul(self) -> None:
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
                    "--landmark-interval-s",
                    "20",
                ]
            )
        for filename in (
            "health_predictions.csv",
            "rul_predictions.csv",
            "baseline_results.json",
            "baseline_manifest.json",
            "checksums.sha256",
        ):
            self.assertEqual(
                (output / filename).read_bytes(),
                (output_copy / filename).read_bytes(),
            )
        results = self._json(output / "baseline_results.json")
        self.assertEqual(results["fit_population"], "train_id only")
        self.assertEqual(results["methods"]["health"], ["constant", "ridge"])
        self.assertEqual(
            results["methods"]["rul"],
            ["naive", "kaplan_meier", "ridge_slope"],
        )
        self.assertGreater(results["kaplan_meier"]["training_trajectories"], 0)
        rul_rows = self._rows(output / "rul_predictions.csv")
        censored = [row for row in rul_rows if row["event_observed"] == "0"]
        self.assertTrue(censored)
        self.assertTrue(all(not row["true_rul_sim_s"] for row in censored))
        self.assertTrue(
            all(float(row["time_to_event_or_censor_sim_s"]) >= 0.0 for row in censored)
        )

    def test_baselines_refuse_train_split_without_observed_eol(self) -> None:
        dataset = self.root / "all-censored-dataset"
        splits = self.root / "all-censored-splits"
        self._run(
            [
                sys.executable,
                str(GENERATE),
                "--output",
                str(dataset),
                "--dataset-version",
                "D3-RUL-all-censored-test",
                "--profiles",
                "1",
                "--trajectories-per-profile",
                "2",
                "--dt-s",
                "1",
                "--sample-interval-s",
                "10",
                "--maximum-sim-duration-s",
                "300",
                "--accelerations",
                "100",
                "--censor-horizons-equivalent-h",
                "8",
                "--seed",
                "999",
            ]
        )
        self._run(
            [
                sys.executable,
                str(SPLIT),
                "--dataset",
                str(dataset),
                "--output",
                str(splits),
            ]
        )
        trajectories = self._rows(dataset / "trajectories.csv")
        self.assertTrue(all(row["event_observed"] == "0" for row in trajectories))
        process = self._run_allow_failure(
            [
                sys.executable,
                str(BASELINES),
                "--dataset",
                str(dataset),
                "--splits",
                str(splits),
                "--output",
                str(self.root / "all-censored-baselines"),
            ]
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("at least one observed EOL in train_id", process.stderr)
        self.assertIn("never used as pseudo-failures", process.stderr)


if __name__ == "__main__":
    unittest.main()
