#!/usr/bin/env python3
"""Generate a self-contained Markdown report from HTST scenario outputs.

The Markdown path has no third-party dependency.  When matplotlib is importable,
the script also writes comparison and time-series figures.  Plotting failures are
reported in the Markdown but never prevent the textual report from being saved.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from provenance import ProvenanceError, verify_checksums


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"
ALARM_PREFIX = "alarm_"
DIAGNOSTIC_ALARM_FIELDS = {"alarm_unsafe_forward", "alarm_high_fouling"}

SCENARIO_LABELS = {
    "normal": "정상 생산",
    "steam_loss": "증기 상실",
    "flow_surge": "유량 급증",
    "sensor_bias_high": "온도센서 고편향(legacy)",
    "control_sensor_bias_high": "제어온도센서 고편향",
    "safety_sensor_bias_high": "안전온도센서 고편향",
    "dual_sensor_common_bias": "두 온도센서 공통 편향",
    "flowmeter_bias_low_with_surge": "유량계 저편향+실유량 급증",
    "booster_pump_failure": "부스터펌프 고장",
    "regenerator_leak_pressure_inversion": "재생부 누설·압력역전",
    "progressive_fouling": "점진 오염",
    "valve_stuck_forward_steam_loss": "FDV 전진고착+증기상실",
    "cooling_utility_loss": "냉각 유틸리티 상실",
    "power_failure": "정전",
    "cip_cycle": "CIP 사이클",
    "start_stop": "기동·정지",
    "incomplete_cleaning": "불완전 CIP",
    "sensor_drift": "온도센서 점진 드리프트",
    "sensor_dropout": "온도센서 드롭아웃",
    "slow_valve": "FDV 저속 응답",
    "valve_leakage": "FDV 내부 누설",
}


class ReportInputError(RuntimeError):
    """Raised when the required report input cannot be parsed."""


@dataclass
class ScenarioReport:
    name: str
    summary: dict[str, Any]
    csv_path: Path | None
    rows: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory containing scenario_summary.json and scenario CSV files",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary JSON path (default: <results-dir>/scenario_summary.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (default: <results-dir>/REPORT.md)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip optional matplotlib figures",
    )
    parser.add_argument(
        "--max-fault-plots",
        type=int,
        default=4,
        help="maximum number of fault dynamics figures (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.max_fault_plots < 0:
        parser.error("--max-fault-plots must be non-negative")
    args.results_dir = args.results_dir.expanduser().resolve()
    args.summary = (args.summary or args.results_dir / "scenario_summary.json").expanduser().resolve()
    args.output = (args.output or args.results_dir / "REPORT.md").expanduser().resolve()
    return args


def read_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise ReportInputError(f"summary JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"cannot parse summary JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError("summary root must be a JSON object")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ReportInputError("summary JSON must contain a 'scenarios' list")
    clean: list[dict[str, Any]] = []
    for index, item in enumerate(scenarios):
        if not isinstance(item, dict) or not isinstance(item.get("scenario"), str):
            raise ReportInputError(f"invalid scenario summary at index {index}")
        clean.append(item)
    return payload, clean


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReportInputError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError(f"{label} root must be a JSON object")
    return payload


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_artifact_provenance(
    results_dir: Path,
    summary_path: Path,
    summary_payload: dict[str, Any],
    scenario_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify checksums and the manifest/summary identity contract."""
    manifest_path = results_dir / "run_manifest.json"
    checksum_path = results_dir / "checksums.sha256"
    manifest = read_json_object(manifest_path, "run manifest")
    try:
        checksums = verify_checksums(results_dir, checksum_path)
    except (OSError, UnicodeError, ProvenanceError) as exc:
        raise ReportInputError(f"artifact checksum verification failed: {exc}") from exc

    raw_artifacts = manifest.get("artifacts")
    if (
        not isinstance(raw_artifacts, list)
        or not all(isinstance(item, str) and item for item in raw_artifacts)
        or len(set(raw_artifacts)) != len(raw_artifacts)
    ):
        raise ReportInputError("run manifest has an invalid or duplicate artifacts list")
    expected_checksums = set(raw_artifacts) | {manifest_path.name}
    if set(checksums) != expected_checksums:
        missing = sorted(expected_checksums - set(checksums))
        extra = sorted(set(checksums) - expected_checksums)
        raise ReportInputError(
            "manifest/checksum artifact set mismatch: "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )

    try:
        summary_relative = summary_path.relative_to(results_dir).as_posix()
    except ValueError as exc:
        raise ReportInputError(
            "summary must be inside results-dir so its checksum can be verified"
        ) from exc
    if summary_relative not in checksums:
        raise ReportInputError(f"summary is not checksummed: {summary_relative}")

    summary_names = [str(item["scenario"]) for item in scenario_summaries]
    if len(set(summary_names)) != len(summary_names):
        raise ReportInputError("summary contains duplicate scenario entries")
    manifest_scenarios = manifest.get("scenarios")
    if not isinstance(manifest_scenarios, list) or not all(
        isinstance(item, str) for item in manifest_scenarios
    ):
        raise ReportInputError("run manifest has an invalid scenarios list")
    if manifest_scenarios != summary_names:
        raise ReportInputError(
            "manifest/summary scenario mismatch: "
            f"manifest={manifest_scenarios}, summary={summary_names}"
        )

    manifest_version = manifest.get("model_version")
    if manifest_version != summary_payload.get("model_version"):
        raise ReportInputError(
            "manifest/summary model_version mismatch: "
            f"{manifest_version!r} != {summary_payload.get('model_version')!r}"
        )
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ReportInputError("run manifest must contain a config object")
    expected_config_hash = config_hash(manifest_config)
    recorded_config_hash = manifest.get("config_hash")
    if recorded_config_hash is not None and recorded_config_hash != expected_config_hash:
        raise ReportInputError(
            "run manifest config_hash does not match its canonical config"
        )
    manifest["_validated_config_hash"] = expected_config_hash
    manifest["_validated_checksums"] = checksums
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReportInputError(f"CSV has no header: {path}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReportInputError(f"cannot parse CSV {path}: {exc}") from exc


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = as_float(mapping.get(key))
        if number is not None:
            return number
    return None


def row_number(row: dict[str, str], *keys: str) -> float | None:
    return first_number(row, *keys)


def numeric_values(rows: Iterable[dict[str, str]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row_number(row, *keys)
        if value is not None:
            values.append(value)
    return values


def sum_column(rows: Iterable[dict[str, str]], *keys: str) -> float | None:
    values = numeric_values(rows, *keys)
    return sum(values) if values else None


def duration_step(row: dict[str, str], default: float) -> float:
    return row_number(row, "step_dt_s") or default


def infer_default_dt(rows: list[dict[str, str]]) -> float:
    explicit = numeric_values(rows[:10], "step_dt_s")
    if explicit:
        return explicit[0]
    times = numeric_values(rows[:10], "time_s")
    diffs = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    return min(diffs) if diffs else 0.0


def is_forward_row(row: dict[str, str]) -> bool:
    forward_l = row_number(row, "forward_l")
    if forward_l is not None:
        return forward_l > 0.0
    return (row_number(row, "fdv_actual_forward") or 0.0) > 0.0


def derive_state_durations(rows: list[dict[str, str]], default_dt: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        mode = row.get("plant_mode")
        if not mode:
            continue
        result[mode] = result.get(mode, 0.0) + duration_step(row, default_dt)
    return dict(sorted(result.items()))


def alarm_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return sorted(
        name
        for name in rows[0]
        if name.startswith(ALARM_PREFIX)
        and name != "alarm_count"
        and name not in DIAGNOSTIC_ALARM_FIELDS
    )


def derive_alarm_metrics(
    rows: list[dict[str, str]], default_dt: float
) -> tuple[dict[str, float], int]:
    columns = alarm_columns(rows)
    durations = {name: 0.0 for name in columns}
    previous = {name: 0 for name in columns}
    activations = 0
    for row in rows:
        step = duration_step(row, default_dt)
        for name in columns:
            state = int((row_number(row, name) or 0.0) > 0.5)
            if state:
                durations[name] += step
            if state and not previous[name]:
                activations += 1
            previous[name] = state
    return ({key: value for key, value in durations.items() if value > 0.0}, activations)


def calculate_metrics(summary: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    default_dt = infer_default_dt(rows)
    forward_rows = [row for row in rows if is_forward_row(row)]
    duration = first_number(summary, "duration_s")
    if duration is None and rows:
        duration = row_number(rows[-1], "time_s")

    forward_l = first_number(summary, "forward_l")
    if forward_l is None:
        forward_l = sum_column(rows, "forward_l")
    routed_l = first_number(summary, "routed_l", "total_feed_l")
    if routed_l is None:
        routed_l = sum_column(rows, "routed_volume_l")
    diverted_l = first_number(summary, "diverted_l")
    if diverted_l is None:
        diverted_l = sum_column(rows, "diverted_l")
    cip_l = first_number(summary, "cip_recirculated_l")
    if cip_l is None:
        cip_l = sum_column(rows, "cip_recirculated_l")
    unsafe_l = first_number(summary, "unsafe_forward_l")
    if unsafe_l is None:
        unsafe_l = sum_column(rows, "unsafe_forward_l")
    quality_oos_l = first_number(summary, "quality_out_of_spec_l")
    if quality_oos_l is None:
        quality_oos_l = sum_column(rows, "quality_out_of_spec_l")

    diverted_fraction = first_number(summary, "diverted_fraction")
    if diverted_fraction is None and routed_l and diverted_l is not None:
        diverted_fraction = diverted_l / routed_l

    min_temp = first_number(summary, "minimum_forward_holding_temp_c")
    if min_temp is None:
        values = numeric_values(forward_rows, "routed_temp_c", "holding_out_temp_c")
        min_temp = min(values) if values else None
    min_residence = first_number(
        summary,
        "minimum_forward_actual_residence_time_s",
        "minimum_forward_residence_time_s",
    )
    if min_residence is None:
        values = numeric_values(
            forward_rows,
            "routed_residence_time_s",
            "actual_residence_time_s",
            "residence_time_s",
        )
        min_residence = min(values) if values else None
    min_lethality = first_number(summary, "minimum_forward_relative_lethality")
    if min_lethality is None:
        values = numeric_values(forward_rows, "routed_relative_lethality", "relative_lethality")
        min_lethality = min(values) if values else None
    min_dp = first_number(summary, "minimum_forward_differential_pressure_bar")
    if min_dp is None:
        values = numeric_values(forward_rows, "differential_pressure_bar")
        min_dp = min(values) if values else None
    max_product_temp = first_number(
        summary,
        "maximum_forward_product_temp_c",
        "maximum_product_temp_c",
    )
    if max_product_temp is None:
        values = numeric_values(
            forward_rows,
            "potential_product_temp_c",
            "product_temp_c",
        )
        max_product_temp = max(values) if values else None

    thermal_energy = first_number(summary, "thermal_energy_kwh")
    if thermal_energy is None:
        thermal_energy = sum_column(rows, "heat_energy_kwh")
    pump_energy = first_number(summary, "pump_energy_kwh")
    if pump_energy is None:
        pump_energy = sum_column(rows, "pump_energy_kwh")
    total_energy = first_number(summary, "total_energy_kwh")
    if total_energy is None and thermal_energy is not None:
        total_energy = thermal_energy + (pump_energy or 0.0)
    specific_energy = first_number(summary, "specific_energy_kwh_per_1000l_forward")
    if specific_energy is None and total_energy is not None and forward_l:
        specific_energy = total_energy / forward_l * 1000.0

    state_durations = summary.get("state_durations_s")
    if not isinstance(state_durations, dict):
        state_durations = derive_state_durations(rows, default_dt)
    else:
        state_durations = {
            str(key): value
            for key, raw in state_durations.items()
            if (value := as_float(raw)) is not None
        }
    alarm_durations = summary.get("alarm_durations_s")
    derived_alarm_durations, derived_activation_count = derive_alarm_metrics(rows, default_dt)
    if not isinstance(alarm_durations, dict):
        alarm_durations = derived_alarm_durations
    else:
        alarm_durations = {
            str(key): value
            for key, raw in alarm_durations.items()
            if (value := as_float(raw)) is not None and value > 0.0
        }
    alarm_activations = first_number(summary, "alarm_activation_count")
    if alarm_activations is None and alarm_columns(rows):
        alarm_activations = float(derived_activation_count)
    total_alarm_seconds = sum(alarm_durations.values()) if alarm_durations else None

    contamination_seconds = 0.0
    has_contamination_signal = False
    fault_active_seconds = 0.0
    has_fault_signal = False
    for row in rows:
        step = duration_step(row, default_dt)
        contamination = row_number(row, "contamination_risk")
        if contamination is not None:
            has_contamination_signal = True
            if contamination > 0.5:
                contamination_seconds += step
        fault = row_number(row, "fault_active")
        if fault is not None:
            has_fault_signal = True
            if fault > 0.5:
                fault_active_seconds += step

    return {
        "duration_s": duration,
        "routed_l": routed_l,
        "forward_l": forward_l,
        "diverted_l": diverted_l,
        "cip_recirculated_l": cip_l,
        "diverted_fraction": diverted_fraction,
        "unsafe_forward_l": unsafe_l,
        "quality_out_of_spec_l": quality_oos_l,
        "minimum_forward_holding_temp_c": min_temp,
        "minimum_forward_residence_time_s": min_residence,
        "minimum_forward_relative_lethality": min_lethality,
        "minimum_forward_differential_pressure_bar": min_dp,
        "maximum_forward_product_temp_c": max_product_temp,
        "thermal_energy_kwh": thermal_energy,
        "pump_energy_kwh": pump_energy,
        "total_energy_kwh": total_energy,
        "specific_energy_kwh_per_1000l_forward": specific_energy,
        "maximum_fouling_index": first_number(summary, "maximum_fouling_index")
        or (max(numeric_values(rows, "fouling_index"), default=None)),
        "final_fouling_index": first_number(summary, "final_fouling_index")
        or (row_number(rows[-1], "fouling_index") if rows else None),
        "final_cip_total_soil_g": first_number(summary, "final_cip_total_soil_g")
        or (row_number(rows[-1], "cip_total_soil_g") if rows else None),
        "final_cip_residual_chemical_fraction": first_number(
            summary, "final_cip_residual_chemical_fraction"
        )
        or (
            row_number(rows[-1], "cip_residual_chemical_fraction")
            if rows
            else None
        ),
        "cip_cleaning_complete": first_number(summary, "cip_cleaning_complete")
        if rows
        else None,
        "state_durations_s": state_durations,
        "alarm_durations_s": alarm_durations,
        "alarm_activation_count": alarm_activations,
        "total_alarm_seconds": total_alarm_seconds,
        "contamination_risk_seconds": contamination_seconds if has_contamination_signal else None,
        "fault_active_seconds": fault_active_seconds if has_fault_signal else None,
        "row_count": len(rows),
        "column_count": len(rows[0]) if rows else 0,
        "default_dt_s": default_dt or None,
    }


def load_scenarios(
    results_dir: Path, scenario_summaries: list[dict[str, Any]]
) -> list[ScenarioReport]:
    csv_paths = {path.stem: path for path in sorted(results_dir.glob("*.csv")) if path.is_file()}
    summary_by_name = {str(item["scenario"]): item for item in scenario_summaries}
    ordered_names = list(summary_by_name)
    # The v1 results directory also contains event_log.csv.  The summary list is
    # authoritative so event/manifest tables cannot be mistaken for scenarios.
    # Header-based discovery is only a fallback for an explicitly empty summary.
    if not ordered_names:
        for name, path in csv_paths.items():
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    fields = set(csv.DictReader(handle).fieldnames or ())
            except (OSError, UnicodeError, csv.Error):
                continue
            if {"scenario", "time_s", "forward_l"}.issubset(fields):
                ordered_names.append(name)
    reports: list[ScenarioReport] = []
    for name in ordered_names:
        summary = summary_by_name.get(name, {"scenario": name})
        csv_path = csv_paths.get(name)
        rows: list[dict[str, str]] = []
        warnings: list[str] = []
        if csv_path is None:
            warnings.append("대응하는 시나리오 CSV가 없어 요약 JSON 값만 사용함")
        else:
            try:
                rows = read_csv(csv_path)
            except ReportInputError as exc:
                warnings.append(str(exc))
        if not rows and csv_path is not None:
            warnings.append("CSV에 데이터 행이 없음")
        report = ScenarioReport(name, summary, csv_path, rows, warnings=warnings)
        report.metrics = calculate_metrics(summary, rows)
        reports.append(report)
    if not reports:
        raise ReportInputError("no scenarios found in summary JSON or results CSV files")
    return reports


def required_row_number(
    row: dict[str, str], key: str, *, scenario: str, index: int
) -> float:
    value = as_float(row.get(key))
    if value is None:
        raise ReportInputError(
            f"{scenario}.csv row {index} has no finite numeric {key!r}"
        )
    return value


def _summary_contract_from_csv(
    scenario: str,
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ReportInputError(f"{scenario}.csv has no data rows")
    previous_end = 0.0
    for index, row in enumerate(rows, start=1):
        start = required_row_number(
            row, "time_start_s", scenario=scenario, index=index
        )
        end = required_row_number(row, "time_s", scenario=scenario, index=index)
        step = required_row_number(
            row, "step_dt_s", scenario=scenario, index=index
        )
        if step <= 0.0 or end <= start:
            raise ReportInputError(
                f"{scenario}.csv row {index} has a non-positive time interval"
            )
        if not math.isclose(end - start, step, rel_tol=1e-9, abs_tol=1e-8):
            raise ReportInputError(
                f"{scenario}.csv row {index} time interval disagrees with step_dt_s"
            )
        if index > 1 and not math.isclose(
            start, previous_end, rel_tol=1e-9, abs_tol=1e-8
        ):
            raise ReportInputError(
                f"{scenario}.csv row {index} is not contiguous with the previous row"
            )
        previous_end = end

    def values(key: str) -> list[float]:
        return [
            required_row_number(row, key, scenario=scenario, index=index)
            for index, row in enumerate(rows, start=1)
        ]

    routed = values("routed_volume_l")
    forward = values("forward_l")
    diverted = values("diverted_l")
    cip = values("cip_recirculated_l")
    unsafe = values("unsafe_forward_l")
    thermal_unsafe = values("thermal_unsafe_forward_l")
    pressure_noncompliant = values("pressure_noncompliant_forward_l")
    contamination_exposed = values("contamination_exposed_forward_l")
    quality_oos = values("quality_out_of_spec_l")
    heat_energy = values("heat_energy_kwh")
    pump_energy = values("pump_energy_kwh")
    total_routed = sum(routed)
    total_forward = sum(forward)
    total_diverted = sum(diverted)
    total_cip = sum(cip)
    total_heat = sum(heat_energy)
    total_pump = sum(pump_energy)
    forward_indexes = [index for index, amount in enumerate(forward) if amount > 0.0]

    def forward_stat(key: str, operation: Any) -> float | None:
        if not forward_indexes:
            return None
        selected = [
            required_row_number(
                rows[index], key, scenario=scenario, index=index + 1
            )
            for index in forward_indexes
        ]
        return operation(selected)

    state_durations: dict[str, float] = collections.defaultdict(float)
    alarm_durations: dict[str, float] = collections.defaultdict(float)
    alarm_names = alarm_columns(rows)
    previous_alarms = {name: 0 for name in alarm_names}
    alarm_activations = 0
    for index, row in enumerate(rows, start=1):
        step = required_row_number(
            row, "step_dt_s", scenario=scenario, index=index
        )
        mode = row.get("plant_mode")
        if not mode:
            raise ReportInputError(f"{scenario}.csv row {index} has no plant_mode")
        state_durations[mode] += step
        for alarm in alarm_names:
            state = int(
                required_row_number(
                    row, alarm, scenario=scenario, index=index
                )
                > 0.5
            )
            if state:
                alarm_durations[alarm] += step
            if state and not previous_alarms[alarm]:
                alarm_activations += 1
            previous_alarms[alarm] = state

    nominal_flow_l_s = float(config["nominal_flow_l_h"]) / 3600.0
    holding_inventory = nominal_flow_l_s * float(config["nominal_holding_time_s"])
    fdv_inventory = nominal_flow_l_s * float(config["sensor_to_fdv_delay_s"])
    final_fouling = values("fouling_index")[-1]
    initial_fouling = float(
        config[
            "cip_initial_fouling_index"
            if scenario in {"cip_cycle", "incomplete_cleaning"}
            else "initial_fouling_index"
        ]
    )
    return {
        "duration_s": required_row_number(
            rows[-1], "time_s", scenario=scenario, index=len(rows)
        ),
        "routed_l": round(total_routed, 3),
        "forward_l": round(total_forward, 3),
        "diverted_l": round(total_diverted, 3),
        "cip_recirculated_l": round(total_cip, 3),
        "diverted_fraction": (
            round(total_diverted / total_routed, 6) if total_routed else 0.0
        ),
        "unsafe_forward_l": round(sum(unsafe), 6),
        "thermal_unsafe_forward_l": round(sum(thermal_unsafe), 6),
        "pressure_noncompliant_forward_l": round(sum(pressure_noncompliant), 6),
        "contamination_exposed_forward_l": round(sum(contamination_exposed), 6),
        "unsafe_forward_fraction_of_forward": (
            round(sum(unsafe) / total_forward, 6) if total_forward else None
        ),
        "quality_out_of_spec_l": round(sum(quality_oos), 6),
        "first_forward_time_s": (
            required_row_number(
                rows[forward_indexes[0]],
                "time_s",
                scenario=scenario,
                index=forward_indexes[0] + 1,
            )
            if forward_indexes
            else None
        ),
        "minimum_forward_holding_temp_c": (
            round(value, 4)
            if (value := forward_stat("forward_temp_c", min)) is not None
            else None
        ),
        "minimum_forward_actual_residence_time_s": (
            round(value, 4)
            if (value := forward_stat("forward_residence_time_s", min)) is not None
            else None
        ),
        "minimum_forward_relative_lethality": (
            round(value, 6)
            if (value := forward_stat("forward_relative_lethality", min)) is not None
            else None
        ),
        "minimum_forward_differential_pressure_bar": (
            round(value, 4)
            if (
                value := forward_stat(
                    "forward_min_process_differential_pressure_bar", min
                )
            )
            is not None
            else None
        ),
        "maximum_forward_product_temp_c": (
            round(value, 4)
            if (value := forward_stat("potential_product_temp_c", max)) is not None
            else None
        ),
        "maximum_fouling_index": round(max(values("fouling_index")), 6),
        "final_fouling_index": round(final_fouling, 6),
        "fouling_reduction_fraction": (
            round(max(0.0, (initial_fouling - final_fouling) / initial_fouling), 6)
            if initial_fouling > 0.0
            else None
        ),
        "thermal_energy_kwh": round(total_heat, 4),
        "pump_energy_kwh": round(total_pump, 4),
        "total_energy_kwh": round(total_heat + total_pump, 4),
        "specific_energy_kwh_per_1000l_forward": (
            round((total_heat + total_pump) / total_forward * 1000.0, 4)
            if total_forward
            else None
        ),
        "maximum_abs_mass_balance_error_l": max(
            abs(value) for value in values("mass_balance_error_l")
        ),
        "maximum_holding_inventory_error_l": round(
            max(
                abs(value - holding_inventory)
                for value in values("holding_inventory_l")
            ),
            9,
        ),
        "maximum_fdv_line_inventory_error_l": round(
            max(
                abs(value - fdv_inventory)
                for value in values("fdv_line_inventory_l")
            ),
            9,
        ),
        "state_durations_s": dict(sorted(state_durations.items())),
        "alarm_durations_s": dict(sorted(alarm_durations.items())),
        "alarm_activation_count": alarm_activations,
    }


def _same_contract_value(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_contract_value(actual[key], expected[key]) for key in actual
        )
    return actual == expected


def validate_scenario_contracts(
    reports: list[ScenarioReport], manifest: dict[str, Any], results_dir: Path
) -> None:
    config = manifest["config"]
    expected_config_hash = str(manifest["_validated_config_hash"])
    checksums = manifest["_validated_checksums"]
    artifacts = set(manifest["artifacts"])
    manifest_run_ids = manifest.get("run_ids")
    if manifest_run_ids is not None and not isinstance(manifest_run_ids, dict):
        raise ReportInputError("run manifest run_ids must be an object")

    validated_run_ids: dict[str, str] = {}
    for report in reports:
        if report.csv_path is None or not report.rows:
            raise ReportInputError(
                f"verified v1 report requires a non-empty CSV for {report.name}"
            )
        relative = report.csv_path.relative_to(results_dir).as_posix()
        if relative not in checksums or relative not in artifacts:
            raise ReportInputError(f"scenario CSV is not declared and checksummed: {relative}")

        scenarios = {row.get("scenario") for row in report.rows}
        run_ids = {row.get("run_id") for row in report.rows}
        config_hashes = {row.get("config_hash") for row in report.rows}
        versions = {row.get("model_version") for row in report.rows}
        if scenarios != {report.name}:
            raise ReportInputError(
                f"{relative} scenario identity mismatch: {sorted(str(x) for x in scenarios)}"
            )
        if len(run_ids) != 1 or None in run_ids or "" in run_ids:
            raise ReportInputError(f"{relative} contains inconsistent run_id values")
        if config_hashes != {expected_config_hash}:
            raise ReportInputError(f"{relative} config_hash does not match the manifest config")
        if versions != {manifest.get("model_version")}:
            raise ReportInputError(f"{relative} model_version does not match the manifest")
        run_id = str(next(iter(run_ids)))
        validated_run_ids[report.name] = run_id

        summary = report.summary
        if summary.get("scenario") != report.name:
            raise ReportInputError(f"summary scenario mismatch for {report.name}")
        if summary.get("run_id") != run_id:
            raise ReportInputError(f"summary/CSV run_id mismatch for {report.name}")
        if summary.get("config_hash") != expected_config_hash:
            raise ReportInputError(f"summary/CSV config_hash mismatch for {report.name}")
        if summary.get("model_version") != manifest.get("model_version"):
            raise ReportInputError(f"summary model_version mismatch for {report.name}")
        if summary.get("config") != config:
            raise ReportInputError(f"summary/manifest config mismatch for {report.name}")
        if isinstance(manifest_run_ids, dict) and manifest_run_ids.get(report.name) != run_id:
            raise ReportInputError(f"manifest/CSV run_id mismatch for {report.name}")

        expected_summary = _summary_contract_from_csv(report.name, report.rows, config)
        for key, expected in expected_summary.items():
            if key not in summary:
                raise ReportInputError(
                    f"summary is missing CSV-derived field {key!r} for {report.name}"
                )
            if not _same_contract_value(summary[key], expected):
                raise ReportInputError(
                    f"summary/CSV mismatch for {report.name}.{key}: "
                    f"summary={summary[key]!r}, CSV-derived={expected!r}"
                )

    event_name = "event_log.csv"
    if event_name in artifacts:
        event_rows = read_csv(results_dir / event_name)
        for index, row in enumerate(event_rows, start=1):
            scenario = row.get("scenario")
            if scenario not in validated_run_ids:
                raise ReportInputError(
                    f"event_log.csv row {index} has unknown scenario {scenario!r}"
                )
            if row.get("run_id") != validated_run_ids[scenario]:
                raise ReportInputError(
                    f"event_log.csv row {index} run_id does not match {scenario}"
                )


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(md_escape(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(md_escape(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000_000:
        text = f"{number:,.0f}"
    elif abs(number) >= 100:
        text = f"{number:,.1f}"
    else:
        text = f"{number:,.{digits}f}"
    return text + suffix


def fmt_fraction(value: Any) -> str:
    number = as_float(value)
    return "—" if number is None else f"{number * 100:.2f}%"


def fmt_duration(value: Any) -> str:
    seconds = as_float(value)
    if seconds is None:
        return "—"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} s"


def scenario_title(name: str) -> str:
    return f"{SCENARIO_LABELS.get(name, name)} (`{name}`)"


def compact_mapping(mapping: dict[str, float], limit: int = 4) -> str:
    if not mapping:
        return "미제공"
    ordered = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{key} {fmt_duration(value)}" for key, value in ordered[:limit]]
    if len(ordered) > limit:
        parts.append(f"외 {len(ordered) - limit}개")
    return "; ".join(parts)


def safety_status(item: ScenarioReport) -> str:
    metrics = item.metrics
    unsafe = as_float(metrics.get("unsafe_forward_l"))
    contamination = as_float(metrics.get("contamination_risk_seconds"))
    if unsafe is not None and unsafe > 1e-9:
        return "비안전 전진 발생"
    if contamination is not None and contamination > 0.0:
        return "오염위험 신호 발생"
    if unsafe is None:
        return "판정자료 부족"
    return "모델상 비안전 전진 없음"


def image_link(report_path: Path, image_path: Path) -> str:
    relative = os.path.relpath(image_path, report_path.parent)
    return Path(relative).as_posix()


def plot_series(rows: list[dict[str, str]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row_number(row, *keys)
        values.append(value if value is not None else math.nan)
    return values


def has_finite(values: Sequence[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def downsample_rows(rows: list[dict[str, str]], maximum: int = 4000) -> list[dict[str, str]]:
    if len(rows) <= maximum:
        return rows
    stride = math.ceil(len(rows) / maximum)
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def short_label(name: str, maximum: int = 18) -> str:
    return name if len(name) <= maximum else name[: maximum - 1] + "…"


def plot_comparison(plt: Any, scenarios: list[ScenarioReport], output: Path) -> None:
    labels = [short_label(item.name) for item in scenarios]
    panels = [
        ("Unsafe forward", "unsafe_forward_l", 1.0, "L", "#c62828"),
        ("Diversion", "diverted_fraction", 100.0, "%", "#ef6c00"),
        ("Quality out-of-spec", "quality_out_of_spec_l", 1.0, "L", "#8e24aa"),
        ("Max product temperature", "maximum_forward_product_temp_c", 1.0, "°C", "#0277bd"),
        ("Total energy", "total_energy_kwh", 1.0, "kWh", "#2e7d32"),
        ("Alarm activations", "alarm_activation_count", 1.0, "count", "#455a64"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(max(14, len(labels) * 1.2), 9))
    for axis, (title, key, scale, unit, color) in zip(axes.flat, panels):
        raw_values = [as_float(item.metrics.get(key)) for item in scenarios]
        if all(value is None for value in raw_values):
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "Not available\nin this result schema",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#616161",
            )
            axis.set_axis_off()
            continue
        values = [value * scale if value is not None else math.nan for value in raw_values]
        axis.bar(range(len(labels)), values, color=color, alpha=0.85)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("HTST scenario comparison — unvalidated surrogate", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def add_fault_spans(axes: Iterable[Any], rows: list[dict[str, str]], times: list[float]) -> None:
    fault = plot_series(rows, "fault_active")
    if not fault or not has_finite(fault):
        return
    start: float | None = None
    intervals: list[tuple[float, float]] = []
    for time_s, active in zip(times, fault):
        is_active = math.isfinite(active) and active > 0.5
        if is_active and start is None:
            start = time_s
        elif not is_active and start is not None:
            intervals.append((start, time_s))
            start = None
    if start is not None and times:
        intervals.append((start, times[-1]))
    for axis in axes:
        for start_s, end_s in intervals:
            axis.axvspan(start_s, end_s, color="#ef5350", alpha=0.10)


def plot_dynamics(plt: Any, item: ScenarioReport, output: Path) -> None:
    rows = downsample_rows(item.rows)
    if not rows:
        raise ValueError("no CSV rows available")
    times = plot_series(rows, "time_s")
    if not has_finite(times):
        raise ValueError("CSV has no numeric time_s column")
    figure, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    temperature_series = [
        ("holding true", ("routed_temp_c", "holding_out_temp_c"), "#d32f2f"),
        ("control sensor", ("control_temp_sensor_c", "sensor_temp_c"), "#1976d2"),
        ("safety sensor", ("safety_temp_sensor_c",), "#7b1fa2"),
        ("product", ("potential_product_temp_c", "product_temp_c"), "#388e3c"),
    ]
    for label, keys, color in temperature_series:
        values = plot_series(rows, *keys)
        if has_finite(values):
            axes[0].plot(times, values, label=label, color=color, linewidth=1.2)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="best", ncol=2, fontsize=8)

    true_flow = plot_series(rows, "flow_l_h")
    measured_flow = plot_series(rows, "measured_flow_l_h")
    if has_finite(true_flow):
        axes[1].plot(times, true_flow, label="true flow", color="#00695c")
    if has_finite(measured_flow):
        axes[1].plot(times, measured_flow, label="measured flow", color="#26a69a", linestyle="--")
    axes[1].set_ylabel("Flow (L/h)")
    residence_axis = axes[1].twinx()
    residence = plot_series(
        rows,
        "actual_residence_time_s",
        "routed_residence_time_s",
        "residence_time_s",
    )
    if has_finite(residence):
        residence_axis.plot(times, residence, label="residence", color="#f9a825", alpha=0.85)
        residence_axis.set_ylabel("Residence (s)")
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = residence_axis.get_legend_handles_labels()
    if handles or handles2:
        axes[1].legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

    differential = plot_series(rows, "differential_pressure_bar")
    measured_differential = plot_series(rows, "measured_differential_pressure_bar")
    if has_finite(differential):
        axes[2].plot(times, differential, label="true ΔP", color="#5d4037")
    if has_finite(measured_differential):
        axes[2].plot(times, measured_differential, label="measured ΔP", color="#8d6e63", linestyle="--")
    axes[2].set_ylabel("Differential pressure (bar)")
    actuator_axis = axes[2].twinx()
    steam = plot_series(rows, "steam_valve")
    fouling = plot_series(rows, "fouling_index")
    if has_finite(steam):
        actuator_axis.plot(times, steam, label="steam valve", color="#fb8c00", alpha=0.8)
    if has_finite(fouling):
        actuator_axis.plot(times, fouling, label="fouling", color="#6d4c41", alpha=0.9)
    actuator_axis.set_ylabel("Valve / fouling (0–1)")
    handles, labels = axes[2].get_legend_handles_labels()
    handles2, labels2 = actuator_axis.get_legend_handles_labels()
    if handles or handles2:
        axes[2].legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

    discrete_series = [
        ("FDV command", ("fdv_command_forward",), "#1565c0"),
        ("FDV actual", ("fdv_actual_forward",), "#2e7d32"),
        ("fault active", ("fault_active",), "#c62828"),
        ("alarm count", ("alarm_count",), "#6a1b9a"),
    ]
    for label, keys, color in discrete_series:
        values = plot_series(rows, *keys)
        if has_finite(values):
            axes[3].step(times, values, where="post", label=label, color=color, linewidth=1.1)
    axes[3].set_ylabel("State / count")
    axes[3].set_xlabel("Simulation time (s)")
    axes[3].legend(loc="best", ncol=2, fontsize=8)

    add_fault_spans(axes, rows, times)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{item.name} dynamics — unvalidated surrogate", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "scenario"


def fault_rank(item: ScenarioReport) -> tuple[float, ...]:
    metrics = item.metrics
    return (
        1.0 if (as_float(metrics.get("unsafe_forward_l")) or 0.0) > 0.0 else 0.0,
        as_float(metrics.get("unsafe_forward_l")) or 0.0,
        1.0 if (as_float(metrics.get("contamination_risk_seconds")) or 0.0) > 0.0 else 0.0,
        as_float(metrics.get("contamination_risk_seconds")) or 0.0,
        as_float(metrics.get("quality_out_of_spec_l")) or 0.0,
        as_float(metrics.get("alarm_activation_count")) or 0.0,
        as_float(metrics.get("diverted_fraction")) or 0.0,
        as_float(metrics.get("total_energy_kwh")) or 0.0,
    )


def generate_plots(
    scenarios: list[ScenarioReport], results_dir: Path, maximum_faults: int
) -> tuple[list[tuple[str, Path]], list[str]]:
    figures: list[tuple[str, Path]] = []
    notes: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # optional dependency; Markdown must still succeed
        notes.append(f"matplotlib 사용 불가로 그래프 생성을 생략함: {type(exc).__name__}: {exc}")
        return figures, notes

    comparison_path = results_dir / "scenario_comparison.png"
    try:
        plot_comparison(plt, scenarios, comparison_path)
        figures.append(("시나리오 핵심지표 비교", comparison_path))
    except Exception as exc:  # keep report generation independent from plotting
        notes.append(f"시나리오 비교 그래프 생성 실패: {type(exc).__name__}: {exc}")

    normal = next((item for item in scenarios if item.name == "normal" and item.rows), None)
    if normal is not None:
        path = results_dir / "normal_dynamics.png"
        try:
            plot_dynamics(plt, normal, path)
            figures.append(("정상 생산 동역학", path))
        except Exception as exc:
            notes.append(f"정상 동역학 그래프 생성 실패: {type(exc).__name__}: {exc}")
    else:
        notes.append("normal.csv가 없어 정상 동역학 그래프를 생성하지 못함")

    candidates = sorted(
        (item for item in scenarios if item.name != "normal" and item.rows),
        key=fault_rank,
        reverse=True,
    )[:maximum_faults]
    for item in candidates:
        path = results_dir / f"{slug(item.name)}_dynamics.png"
        try:
            plot_dynamics(plt, item, path)
            figures.append((f"주요고장 동역학: {scenario_title(item.name)}", path))
        except Exception as exc:
            notes.append(f"{item.name} 동역학 그래프 생성 실패: {type(exc).__name__}: {exc}")
    return figures, notes


def report_insights(scenarios: list[ScenarioReport]) -> list[str]:
    def maximum(key: str) -> ScenarioReport | None:
        candidates = [item for item in scenarios if as_float(item.metrics.get(key)) is not None]
        return max(candidates, key=lambda item: as_float(item.metrics[key]) or 0.0) if candidates else None

    unsafe = maximum("unsafe_forward_l")
    diversion = maximum("diverted_fraction")
    product_temp = maximum("maximum_forward_product_temp_c")
    energy = maximum("total_energy_kwh")
    alarms = maximum("alarm_activation_count")
    lines: list[str] = []
    if unsafe is not None:
        amount = as_float(unsafe.metrics.get("unsafe_forward_l")) or 0.0
        if amount > 0.0:
            lines.append(f"모델상 비안전 전진량이 가장 큰 시나리오는 {scenario_title(unsafe.name)}로 {fmt_number(amount)} L다.")
        else:
            lines.append("현재 요약에서 모델상 비안전 전진량은 모든 시나리오가 0 L다.")
    if diversion is not None:
        lines.append(
            f"회송 비율이 가장 큰 시나리오는 {scenario_title(diversion.name)}로 "
            f"{fmt_fraction(diversion.metrics.get('diverted_fraction'))}다."
        )
    if product_temp is not None:
        lines.append(
            f"최대 제품온도가 가장 높은 시나리오는 {scenario_title(product_temp.name)}로 "
            f"{fmt_number(product_temp.metrics.get('maximum_forward_product_temp_c'))}°C다."
        )
    if energy is not None:
        lines.append(
            f"총 모델 에너지가 가장 큰 시나리오는 {scenario_title(energy.name)}로 "
            f"{fmt_number(energy.metrics.get('total_energy_kwh'))} kWh다."
        )
    if alarms is not None:
        lines.append(
            f"알람 활성화 횟수가 가장 큰 시나리오는 {scenario_title(alarms.name)}로 "
            f"{fmt_number(alarms.metrics.get('alarm_activation_count'), 0)}회다."
        )
    return lines


def build_markdown(
    payload: dict[str, Any],
    summary_path: Path,
    scenarios: list[ScenarioReport],
    report_path: Path,
    figures: list[tuple[str, Path]],
    plot_notes: list[str],
    plots_requested: bool,
) -> str:
    model_status = str(payload.get("model_status", "unvalidated engineering surrogate"))
    versions = sorted(
        {
            str(
                item.summary.get("model_version")
                or (item.rows[0].get("model_version") if item.rows else None)
                or "legacy/unknown"
            )
            for item in scenarios
        }
    )
    lines = [
        "# HTST 시뮬레이션 자동 보고서",
        "",
        "> [!WARNING]",
        "> 이 결과는 **물리·현장 검증 전(unvalidated) 엔지니어링 surrogate**의 출력이다. "
        "실제 미생물 사멸, 법규 적합성, 제품 출하 판정, 설비 안전 인증 또는 실제 위험량으로 사용할 수 없다.",
        "",
        f"- 모델 상태: `{model_status}`",
        f"- 감지된 모델 버전: `{', '.join(versions)}`",
        f"- 요약 입력: `{summary_path.name}`",
        f"- 시나리오 수: {len(scenarios)}",
        "",
        "## 핵심 해석",
        "",
    ]
    insights = report_insights(scenarios)
    lines.extend(f"- {item}" for item in insights)
    if not insights:
        lines.append("- 비교 가능한 수치가 없다.")
    lines.extend(
        [
            "",
            "> 위 순위와 수치는 시뮬레이터 가정 안에서의 상대 비교다. `안전` 표기는 인증 또는 출하 적합 판정을 뜻하지 않는다.",
            "",
            "## 데이터 범위와 상태",
            "",
            md_table(
                ["시나리오", "CSV", "행", "열", "시간간격", "지속시간", "주의"],
                [
                    [
                        scenario_title(item.name),
                        item.csv_path.name if item.csv_path else "없음",
                        item.metrics["row_count"],
                        item.metrics["column_count"],
                        fmt_number(item.metrics.get("default_dt_s"), 3, " s"),
                        fmt_duration(item.metrics.get("duration_s")),
                        "; ".join(item.warnings) if item.warnings else "—",
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "## 안전 비교",
            "",
            md_table(
                [
                    "시나리오",
                    "모델 진단",
                    "비안전 전진(L)",
                    "최저 전진온도(°C)",
                    "최저 체류(s)",
                    "최저 상대열처리",
                    "최저 ΔP(bar)",
                    "오염위험 지속",
                ],
                [
                    [
                        scenario_title(item.name),
                        safety_status(item),
                        fmt_number(item.metrics.get("unsafe_forward_l"), 3),
                        fmt_number(item.metrics.get("minimum_forward_holding_temp_c"), 3),
                        fmt_number(item.metrics.get("minimum_forward_residence_time_s"), 3),
                        fmt_number(item.metrics.get("minimum_forward_relative_lethality"), 5),
                        fmt_number(item.metrics.get("minimum_forward_differential_pressure_bar"), 3),
                        fmt_duration(item.metrics.get("contamination_risk_seconds")),
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "## 품질·생산 비교",
            "",
            md_table(
                [
                    "시나리오",
                    "처리량(L)",
                    "전진량(L)",
                    "회송량(L)",
                    "회송률",
                    "품질이탈량(L)",
                    "최대 제품온도(°C)",
                    "CIP 순환량(L)",
                ],
                [
                    [
                        scenario_title(item.name),
                        fmt_number(item.metrics.get("routed_l")),
                        fmt_number(item.metrics.get("forward_l")),
                        fmt_number(item.metrics.get("diverted_l")),
                        fmt_fraction(item.metrics.get("diverted_fraction")),
                        fmt_number(item.metrics.get("quality_out_of_spec_l"), 3),
                        fmt_number(item.metrics.get("maximum_forward_product_temp_c"), 3),
                        fmt_number(item.metrics.get("cip_recirculated_l")),
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "## 에너지 비교",
            "",
            md_table(
                ["시나리오", "열에너지(kWh)", "펌프(kWh)", "총에너지(kWh)", "전진 1,000L당(kWh)"],
                [
                    [
                        scenario_title(item.name),
                        fmt_number(item.metrics.get("thermal_energy_kwh"), 4),
                        fmt_number(item.metrics.get("pump_energy_kwh"), 4),
                        fmt_number(item.metrics.get("total_energy_kwh"), 4),
                        fmt_number(item.metrics.get("specific_energy_kwh_per_1000l_forward"), 4),
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "## 운전상태 비교",
            "",
            md_table(
                [
                    "시나리오",
                    "상태별 지속시간",
                    "고장 주입 활성시간",
                    "최대 오염도",
                    "최종 오염도",
                    "최종 soil(g)",
                    "잔류 화학분율",
                    "내부 세정판정",
                ],
                [
                    [
                        scenario_title(item.name),
                        compact_mapping(item.metrics.get("state_durations_s") or {}),
                        fmt_duration(item.metrics.get("fault_active_seconds")),
                        fmt_number(item.metrics.get("maximum_fouling_index"), 4),
                        fmt_number(item.metrics.get("final_fouling_index"), 4),
                        fmt_number(item.metrics.get("final_cip_total_soil_g"), 3),
                        fmt_number(
                            item.metrics.get("final_cip_residual_chemical_fraction"),
                            6,
                        ),
                        (
                            "완료"
                            if item.metrics.get("cip_cleaning_complete") == 1.0
                            else "미완료"
                            if item.name in {"cip_cycle", "incomplete_cleaning"}
                            else "—"
                        ),
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "## 알람 비교",
            "",
            md_table(
                ["시나리오", "활성화 횟수", "알람 누적시간*", "주요 활성 알람"],
                [
                    [
                        scenario_title(item.name),
                        fmt_number(item.metrics.get("alarm_activation_count"), 0),
                        fmt_duration(item.metrics.get("total_alarm_seconds")),
                        compact_mapping(item.metrics.get("alarm_durations_s") or {}),
                    ]
                    for item in scenarios
                ],
            ),
            "",
            "\\* 여러 알람이 동시에 활성화되면 누적시간은 실제 경과시간보다 클 수 있다.",
            "",
            "## 그래프",
            "",
        ]
    )
    if figures:
        for title, path in figures:
            lines.extend([f"### {title}", "", f"![{md_escape(title)}]({image_link(report_path, path)})", ""])
    elif plots_requested:
        lines.append("그래프가 생성되지 않았다. 아래 생성 메모를 확인한다.")
        lines.append("")
    else:
        lines.extend(["`--no-plots` 옵션으로 그래프 생성을 생략했다.", ""])
    if plot_notes:
        lines.extend(["### 그래프 생성 메모", ""])
        lines.extend(f"- {note}" for note in plot_notes)
        lines.append("")

    lines.extend(["## 시나리오별 상세", ""])
    for item in scenarios:
        metrics = item.metrics
        lines.extend(
            [
                f"### {scenario_title(item.name)}",
                "",
                f"- 모델 진단: **{safety_status(item)}**",
                f"- 전진/회송: {fmt_number(metrics.get('forward_l'))} L / {fmt_number(metrics.get('diverted_l'))} L",
                f"- 비안전 전진/품질이탈: {fmt_number(metrics.get('unsafe_forward_l'), 3)} L / {fmt_number(metrics.get('quality_out_of_spec_l'), 3)} L",
                f"- 총에너지: {fmt_number(metrics.get('total_energy_kwh'), 4)} kWh",
                f"- 상태: {compact_mapping(metrics.get('state_durations_s') or {})}",
                f"- 알람: {fmt_number(metrics.get('alarm_activation_count'), 0)}회; {compact_mapping(metrics.get('alarm_durations_s') or {})}",
            ]
        )
        if item.csv_path:
            lines.append(f"- 원시 시계열: [{item.csv_path.name}]({image_link(report_path, item.csv_path)})")
        if item.warnings:
            lines.append(f"- 데이터 주의: {'; '.join(item.warnings)}")
        lines.append("")

    lines.extend(
        [
            "## 해석 제한",
            "",
            "- 상대 열처리 지수와 `actual_safe`는 대상 미생물별 검증된 사멸모델이 아니라 시뮬레이션 진단값이다.",
            "- 압력차·누설·오염·CIP·에너지 결과는 코드의 가정과 파라미터 범위 안에서만 의미가 있다.",
            "- 센서 편향과 복합고장 결과는 실제 PLC 배선, 독립 안전기록계, 밸브·펌프 응답을 재현했다고 볼 수 없다.",
            "- 구버전 CSV에 없는 상태·압력·품질·알람 값은 `—` 또는 `미제공`으로 표시하며 0으로 간주하지 않는다.",
            "- 실제 공장에 대한 물리 검증이 제외된 상태이므로 이 보고서를 운전·출하·HACCP 의사결정에 사용하지 않는다.",
            "",
            "---",
            "",
            "이 문서는 `generate_report.py`가 `scenario_summary.json`과 시나리오 CSV에서 자동 생성했다.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload, scenario_summaries = read_summary(args.summary)
        manifest = validate_artifact_provenance(
            args.results_dir,
            args.summary,
            payload,
            scenario_summaries,
        )
        scenarios = load_scenarios(args.results_dir, scenario_summaries)
        validate_scenario_contracts(scenarios, manifest, args.results_dir)
    except ReportInputError as exc:
        print(f"error: {exc}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, Path]] = []
    plot_notes: list[str] = []
    if not args.no_plots:
        figures, plot_notes = generate_plots(scenarios, args.results_dir, args.max_fault_plots)
    markdown = build_markdown(
        payload,
        args.summary,
        scenarios,
        args.output,
        figures,
        plot_notes,
        plots_requested=not args.no_plots,
    )
    try:
        args.output.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write report {args.output}: {exc}")
        return 2
    print(f"report: {args.output}")
    if figures:
        print("figures:")
        for _, path in figures:
            print(f"  {path}")
    for note in plot_notes:
        print(f"plot note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
