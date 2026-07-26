#!/usr/bin/env python3
"""Fast subprocess integration tests for the HTST command-line tools."""

from __future__ import annotations

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

from model import ALARM_COLUMNS, HTSTConfig  # noqa: E402
from uncertainty import (  # noqa: E402
    _fault_window_metrics,
    counterfactual_reference_scenario,
    default_sampling_ranges,
)


RUN_SCENARIOS = MODULE_DIR / "run_scenarios.py"
UNCERTAINTY = MODULE_DIR / "uncertainty.py"
GENERATE_REPORT = MODULE_DIR / "generate_report.py"


class ToolIntegrationTests(unittest.TestCase):
    """Exercise the public CLIs without writing into the repository."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._temporary = tempfile.TemporaryDirectory(prefix="htst-tools-test-")
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.temporary_root = Path(cls._temporary.name)
        cls.scenario_output = cls.temporary_root / "scenario-results"
        cls.scenario_output.mkdir(parents=True)
        (cls.scenario_output / "stale-from-previous-run.csv").write_text(
            "must be removed only after a successful publish\n",
            encoding="utf-8",
        )
        cls.scenario_command = [
            sys.executable,
            str(RUN_SCENARIOS),
            "--scenarios",
            "normal",
            "steam_loss",
            "--duration",
            "90",
            "--dt",
            "1",
            "--seed",
            "24680",
            "--fault-start",
            "40",
            "--fault-duration",
            "20",
            "--severity",
            "1",
            "--output",
            str(cls.scenario_output),
        ]
        cls.scenario_process = cls._run_subprocess(cls.scenario_command)

    @staticmethod
    def _run_subprocess_allow_failure(
        command: Sequence[str],
        *,
        timeout_s: float = 60.0,
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
    def _run_subprocess(
        cls,
        command: Sequence[str],
        *,
        timeout_s: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        process = cls._run_subprocess_allow_failure(command, timeout_s=timeout_s)
        if process.returncode != 0:
            rendered = " ".join(str(part) for part in command)
            raise AssertionError(
                f"command failed with exit {process.returncode}: {rendered}\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )
        return process

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AssertionError(f"CSV has no header: {path}")
            return list(reader)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"JSON root is not an object: {path}")
        return value

    @staticmethod
    def _refresh_checksum(output: Path, relative: str) -> None:
        checksum_path = output / "checksums.sha256"
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        updated: list[str] = []
        found = False
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            _, separator, name = line.partition("  ")
            if separator == "  " and name == relative:
                updated.append(f"{digest}  {relative}")
                found = True
            else:
                updated.append(line)
        if not found:
            raise AssertionError(f"checksum entry not found: {relative}")
        checksum_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def test_run_scenarios_artifact_contract_and_oracle_schema(self) -> None:
        expected_scenarios = {"normal", "steam_loss"}
        expected_files = {
            "normal.csv",
            "steam_loss.csv",
            "event_log.csv",
            "scenario_summary.json",
            "schema.json",
            "run_manifest.json",
            "checksums.sha256",
        }
        actual_files = {
            path.name for path in self.scenario_output.iterdir() if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)
        self.assertFalse(
            (self.scenario_output / "stale-from-previous-run.csv").exists(),
            "publishing a complete run must remove stale files from the target",
        )
        self.assertIn("normal", self.scenario_process.stdout)
        self.assertIn("steam_loss", self.scenario_process.stdout)

        scenario_rows: dict[str, list[dict[str, str]]] = {}
        for scenario in sorted(expected_scenarios):
            rows = self._read_csv(self.scenario_output / f"{scenario}.csv")
            scenario_rows[scenario] = rows
            self.assertTrue(rows)
            self.assertEqual({row["scenario"] for row in rows}, {scenario})
            self.assertEqual(float(rows[0]["time_start_s"]), 0.0)
            self.assertEqual(float(rows[-1]["time_s"]), 90.0)
            self.assertEqual({row["random_seed"] for row in rows}, {"24680"})
            self.assertTrue(all(row["run_id"] for row in rows))
            self.assertTrue(all(row["config_hash"] for row in rows))

        summary = self._read_json(self.scenario_output / "scenario_summary.json")
        self.assertEqual(summary["model_status"], "unvalidated engineering research surrogate")
        self.assertEqual(summary["model_version"], "2.2.0")
        summaries = summary["scenarios"]
        self.assertIsInstance(summaries, list)
        self.assertEqual(
            {item["scenario"] for item in summaries},  # type: ignore[index]
            expected_scenarios,
        )
        for item in summaries:  # type: ignore[assignment]
            self.assertEqual(item["duration_s"], 90.0)
            self.assertIn("maximum_abs_mass_balance_error_l", item)
            self.assertIn("unsafe_forward_l", item)

        event_rows = self._read_csv(self.scenario_output / "event_log.csv")
        self.assertTrue(event_rows)
        self.assertTrue(
            {row["scenario"] for row in event_rows}.issubset(expected_scenarios)
        )
        self.assertTrue(all(row["run_id"] for row in event_rows))
        self.assertTrue(
            all(row["event_type"] in {"MODE", "ALARM"} for row in event_rows)
        )
        self.assertTrue(
            all(row["role"] in {"context", "observable_alarm", "oracle"} for row in event_rows)
        )
        self.assertTrue(any(row["event_type"] == "MODE" for row in event_rows))

        schema = self._read_json(self.scenario_output / "schema.json")
        self.assertEqual(schema["schema_version"], "2.2.0")
        self.assertEqual(schema["model_version"], "2.2.0")
        roles = schema["roles"]
        self.assertIsInstance(roles, dict)
        self.assertIn("forbidden as an ML input", roles["oracle"])  # type: ignore[index]
        fields = schema["fields"]
        self.assertIsInstance(fields, list)
        role_by_name = {
            field["name"]: field["role"]  # type: ignore[index]
            for field in fields  # type: ignore[assignment]
        }
        type_by_name = {
            field["name"]: field["type"]  # type: ignore[index]
            for field in fields  # type: ignore[assignment]
        }
        self.assertEqual(
            set(role_by_name),
            set(scenario_rows["normal"][0]),
            "schema must classify every scenario CSV column",
        )

        expected_oracles = {
            "fault_active",
            "flow_l_h",
            "holding_out_temp_c",
            "actual_residence_time_s",
            "routed_pressure_safe_fraction",
            "forward_min_process_differential_pressure_bar",
            "differential_pressure_bar",
            "fouling_index",
            "actual_safe",
            "heater_capacity_factor",
        }
        for name in expected_oracles:
            with self.subTest(field=name):
                self.assertEqual(role_by_name[name], "oracle")
        for name in (
            "measured_flow_l_h",
            "control_temp_sensor_c",
            "safety_temp_sensor_c",
            "raw_pressure_sensor_bar",
            "leak_detector_signal_fraction",
            "product_temp_sensor_c",
        ):
            with self.subTest(observable=name):
                self.assertEqual(role_by_name[name], "observable_signal")
        self.assertEqual(role_by_name["fdv_actual_forward"], "oracle")
        self.assertEqual(role_by_name["scenario"], "context")
        self.assertEqual(role_by_name["alarm_low_temperature"], "observable_alarm")
        self.assertEqual(
            role_by_name["alarm_count"],
            "observable_alarm",
            "alarm_count contains observable alarms only in v2",
        )
        self.assertEqual(role_by_name["observable_alarm_count"], "observable_alarm")
        self.assertEqual(role_by_name["alarm_unsafe_forward"], "oracle")
        self.assertEqual(role_by_name["alarm_high_fouling"], "oracle")
        self.assertEqual(role_by_name["cip_total_soil_g"], "oracle")
        self.assertEqual(
            role_by_name["fdv_position_feedback"], "observable_signal"
        )
        self.assertEqual(role_by_name["power_available"], "oracle")
        self.assertEqual(role_by_name["sensor_available"], "oracle")
        self.assertEqual(role_by_name["power_good_signal"], "observable_signal")
        self.assertEqual(
            role_by_name["temperature_sensor_quality_ok"],
            "observable_signal",
        )
        self.assertEqual(role_by_name["cip_cycle_active"], "control")
        self.assertEqual(role_by_name["unsafe_forward_l"], "outcome")
        self.assertEqual(type_by_name["quality_out_of_spec_l"], "number")

        manifest = self._read_json(self.scenario_output / "run_manifest.json")
        self.assertEqual(manifest["model_version"], "2.2.0")
        self.assertEqual(set(manifest["scenarios"]), expected_scenarios)  # type: ignore[arg-type]
        config = manifest["config"]
        self.assertIsInstance(config, dict)
        self.assertEqual(config["duration_s"], 90.0)  # type: ignore[index]
        self.assertEqual(config["dt_s"], 1.0)  # type: ignore[index]
        self.assertEqual(config["random_seed"], 24680)  # type: ignore[index]
        self.assertIn("--fault-start 40.0", manifest["reproduce"])
        self.assertRegex(manifest["config_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(set(manifest["run_ids"]), expected_scenarios)
        source = manifest["source_provenance"]
        self.assertEqual(source["algorithm"], "sha256")
        self.assertRegex(source["aggregate_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            set(source["files"]),
            {"model.py", "process_components.py", "provenance.py", "run_scenarios.py"},
        )
        self.assertEqual(
            set(manifest["artifacts"]),  # type: ignore[arg-type]
            expected_files - {"run_manifest.json", "checksums.sha256"},
        )

        checksum_lines = (
            self.scenario_output / "checksums.sha256"
        ).read_text(encoding="utf-8").splitlines()
        checksums: dict[str, str] = {}
        for line in checksum_lines:
            digest, separator, relative = line.partition("  ")
            self.assertEqual(separator, "  ")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertNotIn(relative, checksums)
            checksums[relative] = digest
        self.assertEqual(
            set(checksums),
            expected_files - {"checksums.sha256"},
            "every generated artifact except the checksum file itself must be hashed",
        )
        for relative, expected_digest in checksums.items():
            path = self.scenario_output / relative
            self.assertTrue(path.is_file())
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual_digest, expected_digest)

    def test_runner_rejects_duplicates_and_preserves_existing_output_on_failure(self) -> None:
        output = self.temporary_root / "preserved-runner-output"
        output.mkdir()
        sentinel = output / "existing-result.txt"
        sentinel.write_text("keep me\n", encoding="utf-8")

        duplicate = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(RUN_SCENARIOS),
                "--scenarios",
                "normal",
                "normal",
                "--duration",
                "10",
                "--output",
                str(output),
            ]
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("must not contain duplicates", duplicate.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")

        nonoverlap = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(RUN_SCENARIOS),
                "--scenarios",
                "steam_loss",
                "--duration",
                "10",
                "--fault-start",
                "20",
                "--fault-duration",
                "5",
                "--output",
                str(output),
            ]
        )
        self.assertNotEqual(nonoverlap.returncode, 0)
        self.assertIn("fault interval does not overlap", nonoverlap.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")

        staged_failure = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(RUN_SCENARIOS),
                "--scenarios",
                "normal",
                "--duration",
                "1e-13",
                "--output",
                str(output),
            ]
        )
        self.assertNotEqual(staged_failure.returncode, 0)
        self.assertIn("produced no records", staged_failure.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")
        self.assertFalse(
            list(output.parent.glob(f".{output.name}.staging-*")),
            "failed staging directories must be cleaned up",
        )

    def test_fault_alarm_latency_uses_new_observable_edge_at_interval_start(self) -> None:
        base: dict[str, float | int | str] = {
            name: 0 for name in ALARM_COLUMNS
        }
        base.update(
            {
                "forward_l": 0.0,
                "diverted_l": 0.0,
                "unsafe_forward_l": 0.0,
                "quality_out_of_spec_l": 0.0,
                "step_dt_s": 1.0,
            }
        )
        records = [
            {
                **base,
                "time_start_s": 0.0,
                "time_s": 1.0,
                "fault_active": 0,
                "alarm_low_temperature": 1,
            },
            {
                **base,
                "time_start_s": 1.0,
                "time_s": 2.0,
                "fault_active": 1,
                "alarm_low_temperature": 1,
            },
            {
                **base,
                "time_start_s": 2.0,
                "time_s": 3.0,
                "fault_active": 1,
                "alarm_unsafe_forward": 1,
            },
            {
                **base,
                "time_start_s": 3.0,
                "time_s": 4.0,
                "fault_active": 1,
                "alarm_high_flow": 1,
            },
        ]
        metrics = _fault_window_metrics(
            records,
            HTSTConfig(fault_start_s=1.0, fault_duration_s=3.0),
        )
        self.assertEqual(metrics["first_fault_alarm_latency_s"], 2.0)
        self.assertEqual(metrics["fault_window_alarm_duration_s"], 2.0)
        self.assertEqual(metrics["fault_window_max_alarm_count"], 1)

    def test_uncertainty_rejects_invalid_joint_ranges_before_replacing_output(self) -> None:
        output = self.temporary_root / "preserved-uncertainty-output"
        output.mkdir()
        sentinel = output / "existing-result.txt"
        sentinel.write_text("keep uncertainty\n", encoding="utf-8")
        process = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(UNCERTAINTY),
                "--runs",
                "1",
                "--scenarios",
                "steam_loss",
                "--duration",
                "60",
                "--dt",
                "1",
                "--range",
                "pasteurization_setpoint_c",
                "73.85",
                "74",
                "--range",
                "forward_temperature_margin_c",
                "0.1",
                "1.9",
                "--output",
                str(output),
            ]
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("invalid joint range corner", process.stderr)
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"), "keep uncertainty\n"
        )

    def test_uncertainty_artifacts_and_byte_reproducibility(self) -> None:
        self.assertEqual(
            counterfactual_reference_scenario("incomplete_cleaning"), "cip_cycle"
        )
        self.assertEqual(counterfactual_reference_scenario("steam_loss"), "normal")
        output = self.temporary_root / "uncertainty-results"
        output.mkdir()
        (output / "stale-analysis.csv").write_text("stale\n", encoding="utf-8")
        command = [
            sys.executable,
            str(UNCERTAINTY),
            "--runs",
            "2",
            "--scenarios",
            "steam_loss",
            "flow_surge",
            "--duration",
            "60",
            "--dt",
            "1",
            "--master-seed",
            "13579",
            "--range",
            "fault_severity",
            "0.7",
            "1.1",
            "--output",
            str(output),
        ]
        expected_files = {
            "monte_carlo_runs.csv",
            "monte_carlo_summary.json",
            "sensitivity.csv",
            "sensitivity.json",
            "uncertainty_manifest.json",
            "checksums.sha256",
        }

        first_process = self._run_subprocess(command)
        self.assertEqual(
            {path.name for path in output.iterdir() if path.is_file()},
            expected_files,
        )
        first_bytes = {name: (output / name).read_bytes() for name in expected_files}

        second_process = self._run_subprocess(command)
        self.assertEqual(first_process.stdout, second_process.stdout)
        self.assertEqual(first_process.stderr, second_process.stderr)
        for name, expected in first_bytes.items():
            with self.subTest(reproducible=name):
                self.assertEqual((output / name).read_bytes(), expected)

        run_rows = self._read_csv(output / "monte_carlo_runs.csv")
        self.assertEqual(len(run_rows), 4)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in run_rows:
            grouped[row["sample_id"]].append(row)
        self.assertEqual(len(grouped), 2)
        for sample_id, rows in grouped.items():
            with self.subTest(sample_id=sample_id):
                self.assertEqual(
                    {row["scenario"] for row in rows},
                    {"steam_loss", "flow_surge"},
                )
                self.assertEqual(len({row["config_hash"] for row in rows}), 1)
                self.assertEqual(len({row["random_seed"] for row in rows}), 1)
                self.assertEqual(len({row["sampling_seed"] for row in rows}), 1)
                self.assertEqual(
                    {row["counterfactual_reference_scenario"] for row in rows},
                    {"normal"},
                )
                self.assertEqual(
                    len({row["counterfactual_reference_run_id"] for row in rows}),
                    1,
                )
                for row in rows:
                    self.assertAlmostEqual(
                        float(row["delta_vs_counterfactual_quality_out_of_spec_l"]),
                        float(row["quality_out_of_spec_l"])
                        - float(row["counterfactual_quality_out_of_spec_l"]),
                    )

        monte_carlo = self._read_json(output / "monte_carlo_summary.json")
        self.assertEqual(monte_carlo["analysis"], "monte_carlo")
        self.assertEqual(monte_carlo["model_version"], "2.2.0")
        self.assertEqual(monte_carlo["master_seed"], 13579)
        self.assertEqual(monte_carlo["runs_per_scenario"], 2)
        self.assertEqual(monte_carlo["scenario_simulations"], 4)
        self.assertEqual(monte_carlo["additional_counterfactual_simulations"], 2)
        self.assertEqual(monte_carlo["total_simulations"], 6)
        self.assertEqual(
            set(monte_carlo["scenarios"]),  # type: ignore[arg-type]
            {"steam_loss", "flow_surge"},
        )
        ranges = monte_carlo["sampling_ranges"]
        self.assertIsInstance(ranges, dict)
        self.assertEqual(ranges["fault_severity"]["low"], 0.7)  # type: ignore[index]
        self.assertEqual(ranges["fault_severity"]["high"], 1.1)  # type: ignore[index]
        scenario_results = monte_carlo["scenario_results"]
        self.assertIsInstance(scenario_results, dict)
        for scenario in ("steam_loss", "flow_surge"):
            result = scenario_results[scenario]  # type: ignore[index]
            self.assertIn("metrics", result)
            self.assertIn("incidence", result)
            self.assertIn("counterfactual_deltas", result)

        sensitivity_rows = self._read_csv(output / "sensitivity.csv")
        self.assertTrue(sensitivity_rows)
        self.assertTrue(
            {
                "scenario",
                "parameter",
                "metric",
                "low_metric",
                "baseline_metric",
                "high_metric",
                "scaled_output_change",
                "absolute_effect_rank",
            }.issubset(sensitivity_rows[0])
        )
        sensitivity = self._read_json(output / "sensitivity.json")
        self.assertEqual(sensitivity["analysis"], "one_at_a_time_sensitivity")
        self.assertEqual(sensitivity["model_version"], "2.2.0")
        expected_oat_runs = 2 * (1 + 2 * len(default_sampling_ranges(HTSTConfig())))
        self.assertEqual(sensitivity["simulation_count"], expected_oat_runs)
        self.assertEqual(len(sensitivity["rows"]), len(sensitivity_rows))  # type: ignore[arg-type]
        self.assertIn("rankings", sensitivity)

        manifest = self._read_json(output / "uncertainty_manifest.json")
        self.assertEqual(manifest["model_version"], "2.2.0")
        self.assertEqual(manifest["master_seed"], 13579)
        self.assertEqual(manifest["runs_per_scenario"], 2)
        self.assertEqual(manifest["monte_carlo_scenario_simulations"], 4)
        self.assertEqual(
            manifest["monte_carlo_additional_counterfactual_simulations"], 2
        )
        self.assertEqual(manifest["monte_carlo_total_simulations"], 6)
        self.assertEqual(
            set(manifest["artifacts"]),
            expected_files - {"uncertainty_manifest.json", "checksums.sha256"},
        )
        self.assertIn("--range fault_severity 0.7 1.1", manifest["reproduce"])
        source = manifest["source_provenance"]
        self.assertRegex(source["aggregate_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            set(source["files"]),
            {"model.py", "process_components.py", "provenance.py", "uncertainty.py"},
        )
        checksum_lines = (output / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines()
        checksum_names = {line.partition("  ")[2] for line in checksum_lines}
        self.assertEqual(checksum_names, expected_files - {"checksums.sha256"})
        for line in checksum_lines:
            digest, separator, relative = line.partition("  ")
            self.assertEqual(separator, "  ")
            self.assertEqual(
                hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                digest,
            )

    def test_generate_report_no_plots_warning_and_scenarios(self) -> None:
        report = self.temporary_root / "reports" / "REPORT.md"
        process = self._run_subprocess(
            [
                sys.executable,
                str(GENERATE_REPORT),
                "--results-dir",
                str(self.scenario_output),
                "--output",
                str(report),
                "--no-plots",
            ]
        )
        self.assertTrue(report.is_file())
        self.assertIn(str(report), process.stdout)
        markdown = report.read_text(encoding="utf-8")
        self.assertIn("# HTST 시뮬레이션 자동 보고서", markdown)
        self.assertIn("> [!WARNING]", markdown)
        self.assertIn("unvalidated", markdown)
        self.assertIn("실제 미생물 사멸", markdown)
        self.assertIn("법규 적합성", markdown)
        self.assertIn("`normal`", markdown)
        self.assertIn("`steam_loss`", markdown)
        self.assertIn("정상 생산", markdown)
        self.assertIn("증기 상실", markdown)
        self.assertIn("`--no-plots` 옵션으로 그래프 생성을 생략했다.", markdown)
        self.assertIn("## 해석 제한", markdown)
        self.assertFalse(list(report.parent.glob("*.png")))

    def test_generate_report_rejects_tampered_or_mixed_run_artifacts(self) -> None:
        checksum_copy = self.temporary_root / "tampered-checksum-results"
        shutil.copytree(self.scenario_output, checksum_copy)
        with (checksum_copy / "normal.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        checksum_process = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(GENERATE_REPORT),
                "--results-dir",
                str(checksum_copy),
                "--output",
                str(self.temporary_root / "checksum-failure.md"),
                "--no-plots",
            ]
        )
        self.assertEqual(checksum_process.returncode, 2)
        self.assertIn("checksum mismatch", checksum_process.stdout)

        summary_copy = self.temporary_root / "mixed-summary-results"
        shutil.copytree(self.scenario_output, summary_copy)
        summary_path = summary_copy / "scenario_summary.json"
        summary = self._read_json(summary_path)
        scenarios = summary["scenarios"]
        self.assertIsInstance(scenarios, list)
        scenarios[0]["forward_l"] = float(scenarios[0]["forward_l"]) + 1.0  # type: ignore[index]
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._refresh_checksum(summary_copy, "scenario_summary.json")
        summary_process = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(GENERATE_REPORT),
                "--results-dir",
                str(summary_copy),
                "--output",
                str(self.temporary_root / "summary-failure.md"),
                "--no-plots",
            ]
        )
        self.assertEqual(summary_process.returncode, 2)
        self.assertIn("summary/CSV mismatch", summary_process.stdout)

        identity_copy = self.temporary_root / "mixed-identity-results"
        shutil.copytree(self.scenario_output, identity_copy)
        identity_path = identity_copy / "scenario_summary.json"
        identity = self._read_json(identity_path)
        identity_scenarios = identity["scenarios"]
        self.assertIsInstance(identity_scenarios, list)
        identity_scenarios[0]["run_id"] = "foreign-run"  # type: ignore[index]
        identity_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._refresh_checksum(identity_copy, "scenario_summary.json")
        identity_process = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(GENERATE_REPORT),
                "--results-dir",
                str(identity_copy),
                "--output",
                str(self.temporary_root / "identity-failure.md"),
                "--no-plots",
            ]
        )
        self.assertEqual(identity_process.returncode, 2)
        self.assertIn("summary/CSV run_id mismatch", identity_process.stdout)

        config_copy = self.temporary_root / "mixed-config-results"
        shutil.copytree(self.scenario_output, config_copy)
        config_csv = config_copy / "normal.csv"
        config_rows = self._read_csv(config_csv)
        config_rows[0]["config_hash"] = "0" * 64
        with config_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(config_rows[0]))
            writer.writeheader()
            writer.writerows(config_rows)
        self._refresh_checksum(config_copy, "normal.csv")
        config_process = self._run_subprocess_allow_failure(
            [
                sys.executable,
                str(GENERATE_REPORT),
                "--results-dir",
                str(config_copy),
                "--output",
                str(self.temporary_root / "config-failure.md"),
                "--no-plots",
            ]
        )
        self.assertEqual(config_process.returncode, 2)
        self.assertIn("config_hash does not match", config_process.stdout)


if __name__ == "__main__":
    unittest.main()
