#!/usr/bin/env python3
"""Run the integrated public-reference HTST digital-twin research stack.

The governing process remains ``model.py`` v2.2.  This v3 runner composes the
reference P&ID validator, calibration/sensor layer, vendor-neutral PLC shadow,
HACCP evidence ledger and independent multi-cycle lifecycle model.  It never
drives physical equipment and intentionally contains no biological model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from haccp import EvidenceLedger, load_haccp_plan
from lifecycle import (
    EVENT_FIELDS as LIFECYCLE_EVENT_FIELDS,
    SERVICE_INTERVAL_FIELDS,
    TRACE_FIELDS as LIFECYCLE_TRACE_FIELDS,
    load_maintenance_policy,
    simulate_lifecycle,
)
from model import CampaignPhase, HTSTConfig, HTSTSimulator, MODEL_VERSION, extract_events
from plc_logic import PLCSettings, ReferencePLC
from provenance import source_provenance, staged_output_directory, write_checksums
from reference_plant import load_reference_plant, render_mermaid, validate_reference_plant
from sensor_calibration import (
    CalibrationPoint,
    SensorChain,
    create_calibration_record,
    evaluate_guarded_reading,
    load_sensor_catalog,
)


HERE = Path(__file__).resolve().parent
DIGITAL_TWIN_VERSION = "3.0.0"
CONFIG_SCHEMA_VERSION = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "1.0.0"
DEFAULT_CONFIG = HERE / "digital_twin_config.json"
DEFAULT_OUTPUT = HERE / "digital_twin_results"
MODEL_STATUS = "unvalidated_public_reference_research_surrogate"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_SENSOR_TRUTH_FIELDS = {
    "TT-104": "holding_out_temp_c",
    "FT-101": "flow_l_h",
    "PDT-101": "differential_pressure_bar",
    "AIT-201": "cip_conductivity_proxy_ms_cm",
    "AIT-202": "cip_ph_proxy",
    "AIT-203": "restart_product_interface_signal_fraction",
}
_CORRECTED_PROCESS_FIELDS = {
    "TT-104": "safety_temp_sensor_c",
    "FT-101": "measured_flow_l_h",
    "PDT-101": "measured_differential_pressure_bar",
}

SENSOR_TRACE_FIELDS = (
    "digital_twin_run_id",
    "campaign_time_s",
    "phase_id",
    "sensor_tag",
    "source_field",
    "true_value",
    "filtered_value",
    "indicated_value",
    "quality",
    "dropout",
    "held_last",
    "calibration_record_id",
    "correction_status",
    "corrected_value",
    "combined_standard_uncertainty",
    "expanded_uncertainty",
    "lower_bound",
    "upper_bound",
    "guard_status",
    "guard_disposition",
)

PLC_TRACE_FIELDS = (
    "digital_twin_run_id",
    "campaign_time_s",
    "phase_id",
    "scan_index",
    "previous_state",
    "plc_state",
    "state_changed",
    "cip_recipe_state",
    "forward_permissive",
    "trip_latched",
    "active_causes",
    "active_trip_causes",
    "active_divert_causes",
    "feed_pump_command",
    "booster_pump_command",
    "steam_enable",
    "fdv_command_forward",
    "fdv_command_divert",
    "cip_pump_command",
    "cip_rinse_valve",
    "cip_alkali_valve",
    "cip_acid_valve",
    "cip_drain_valve",
    "core_fdv_command_forward",
    "shadow_core_fdv_command_match",
)

PLC_EVENT_FIELDS = (
    "digital_twin_run_id",
    "campaign_time_s",
    "phase_id",
    "event_type",
    "previous_state",
    "plc_state",
    "active_causes",
)

TIMESERIES_FIELDS = (
    "digital_twin_run_id",
    "campaign_time_start_s",
    "campaign_time_s",
    "phase_id",
    "phase_kind",
    "scenario",
    "plant_mode",
    "control_temp_sensor_c",
    "safety_temp_sensor_c",
    "calibrated_safety_temp_c",
    "measured_flow_l_h",
    "calibrated_flow_l_h",
    "estimated_fastest_residence_time_s",
    "calibrated_fastest_residence_time_s",
    "measured_differential_pressure_bar",
    "calibrated_differential_pressure_bar",
    "fdv_command_forward",
    "fdv_actual_forward",
    "plc_state",
    "plc_fdv_command_forward",
    "plc_active_causes",
    "calibration_valid_count",
    "calibration_unknown_count",
    "haccp_overall_status",
    "haccp_lot_status",
    "product_boundary_l",
    "unsafe_forward_l",
    "quality_out_of_spec_l",
    "fouling_index",
    "cip_total_soil_g",
    "cip_release_permissive",
)

CALIBRATION_SUMMARY_FIELDS = (
    "sensor_tag",
    "record_id",
    "calibrated_at",
    "due_at",
    "as_found_intercept",
    "as_found_slope",
    "as_left_intercept",
    "as_left_slope",
    "combined_base_standard_uncertainty",
    "coverage_factor",
    "expanded_base_uncertainty",
    "status_at_run_start",
    "claim_boundary",
)

HACCP_DEVIATION_FIELDS = (
    "record_id",
    "sequence",
    "time_s",
    "lot_id",
    "record_type",
    "control_id",
    "limit_id",
    "reason",
    "lot_status_after",
    "record_hash",
)


class DigitalTwinConfigError(ValueError):
    """Raised when the integrated run configuration is ambiguous or unsafe."""


def _reject_constant(value: str) -> None:
    raise DigitalTwinConfigError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DigitalTwinConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, object], keys: set[str], context: str) -> None:
    actual = set(value)
    if actual != keys:
        raise DigitalTwinConfigError(
            f"{context} keys mismatch; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DigitalTwinConfigError(f"{context} must be an object")
    return value


def _number(value: object, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DigitalTwinConfigError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise DigitalTwinConfigError(f"{context} must be {qualifier}")
    return result


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DigitalTwinConfigError(f"{context} must be an integer")
    if positive and value <= 0:
        raise DigitalTwinConfigError(f"{context} must be positive")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DigitalTwinConfigError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except DigitalTwinConfigError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise DigitalTwinConfigError(f"cannot load config {path}: {exc}") from exc
    root = dict(_mapping(value, "digital twin config"))
    _exact_keys(
        root,
        {
            "digital_twin_version",
            "config_schema_version",
            "run_id",
            "model_status",
            "seed",
            "process",
            "plc",
            "calibration",
            "haccp",
            "lifecycle",
        },
        "digital twin config",
    )
    if root["digital_twin_version"] != DIGITAL_TWIN_VERSION:
        raise DigitalTwinConfigError("unsupported digital_twin_version")
    if root["config_schema_version"] != CONFIG_SCHEMA_VERSION:
        raise DigitalTwinConfigError("unsupported config_schema_version")
    if root["model_status"] != MODEL_STATUS:
        raise DigitalTwinConfigError(f"model_status must be {MODEL_STATUS!r}")
    if not isinstance(root["run_id"], str) or not _SAFE_ID.fullmatch(root["run_id"]):
        raise DigitalTwinConfigError("run_id has invalid syntax")
    seed = _integer(root["seed"], "seed")
    if not 0 <= seed <= 2**63 - 1:
        raise DigitalTwinConfigError("seed must be in 0..2^63-1")

    process = _mapping(root["process"], "process")
    _exact_keys(
        process,
        {
            "dt_s",
            "initial_fouling_index",
            "process_cycles",
            "production_duration_s",
            "cip_duration_s",
            "restart_duration_s",
            "cip_pattern",
        },
        "process",
    )
    dt_s = _number(process["dt_s"], "process.dt_s", positive=True)
    if not 0.001 <= dt_s <= 1.0:
        raise DigitalTwinConfigError("process.dt_s must be in [0.001, 1.0]")
    fouling = _number(process["initial_fouling_index"], "process.initial_fouling_index")
    if not 0.0 <= fouling <= 1.0:
        raise DigitalTwinConfigError("process.initial_fouling_index must be in [0, 1]")
    cycles = _integer(process["process_cycles"], "process.process_cycles", positive=True)
    for name in ("production_duration_s", "cip_duration_s", "restart_duration_s"):
        _number(process[name], f"process.{name}", positive=True)
    if float(process["cip_duration_s"]) < 540.0:
        raise DigitalTwinConfigError("process.cip_duration_s must cover the 540 s recipe")
    pattern = process["cip_pattern"]
    if (
        not isinstance(pattern, list)
        or not pattern
        or not all(item in {"complete", "incomplete"} for item in pattern)
    ):
        raise DigitalTwinConfigError(
            "process.cip_pattern must be a non-empty list of complete/incomplete"
        )
    if len(pattern) > cycles:
        raise DigitalTwinConfigError("process.cip_pattern cannot exceed process_cycles")

    plc = _mapping(root["plc"], "plc")
    _exact_keys(plc, {"scan_time_s", "forward_confirmation_s", "shadow_only"}, "plc")
    scan_time = _number(plc["scan_time_s"], "plc.scan_time_s", positive=True)
    if not math.isclose(scan_time, dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise DigitalTwinConfigError("plc.scan_time_s must equal process.dt_s")
    _number(plc["forward_confirmation_s"], "plc.forward_confirmation_s", positive=True)
    if plc["shadow_only"] is not True:
        raise DigitalTwinConfigError("plc.shadow_only must be true")

    calibration = _mapping(root["calibration"], "calibration")
    _exact_keys(
        calibration,
        {"evaluation_date", "guard_band_enabled", "coverage_factor"},
        "calibration",
    )
    try:
        datetime.fromisoformat(str(calibration["evaluation_date"]))
    except ValueError as exc:
        raise DigitalTwinConfigError("calibration.evaluation_date must be ISO date") from exc
    if calibration["guard_band_enabled"] is not True:
        raise DigitalTwinConfigError("calibration.guard_band_enabled must be true")
    _number(calibration["coverage_factor"], "calibration.coverage_factor", positive=True)

    haccp = _mapping(root["haccp"], "haccp")
    _exact_keys(
        haccp,
        {"plan_id", "lot_id", "jurisdiction_status", "automatic_release"},
        "haccp",
    )
    for name in ("plan_id", "lot_id"):
        if not isinstance(haccp[name], str) or not _SAFE_ID.fullmatch(haccp[name]):
            raise DigitalTwinConfigError(f"haccp.{name} has invalid syntax")
    if haccp["jurisdiction_status"] != "not_assessed":
        raise DigitalTwinConfigError("haccp.jurisdiction_status must be not_assessed")
    if haccp["automatic_release"] is not False:
        raise DigitalTwinConfigError("haccp.automatic_release must be false")

    lifecycle = _mapping(root["lifecycle"], "lifecycle")
    _exact_keys(lifecycle, {"policy_file", "cycles"}, "lifecycle")
    policy_file = lifecycle["policy_file"]
    if not isinstance(policy_file, str) or not policy_file:
        raise DigitalTwinConfigError("lifecycle.policy_file must be a relative file")
    policy_path = PurePosixPath(policy_file)
    if policy_path.is_absolute() or ".." in policy_path.parts:
        raise DigitalTwinConfigError("lifecycle.policy_file must not escape config directory")
    _integer(lifecycle["cycles"], "lifecycle.cycles", positive=True)
    return copy.deepcopy(root)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row)))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty table {path.name}")
        fields = tuple(rows[0])
    expected = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            if list(row) != expected:
                raise ValueError(f"{path.name} row {index} has schema drift")
            normalized: dict[str, object] = {}
            for name, item in row.items():
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError(f"{path.name} row {index} contains non-finite {name}")
                if isinstance(item, (dict, list, tuple)):
                    normalized[name] = _canonical_bytes(item).decode("utf-8").rstrip("\n")
                elif item is None:
                    normalized[name] = ""
                else:
                    normalized[name] = item
            writer.writerow(normalized)


def _calibration_points(specification: Any) -> tuple[list[CalibrationPoint], list[CalibrationPoint]]:
    low = float(specification.minimum)
    high = float(specification.maximum)
    middle = 0.5 * (low + high)
    references = (low, middle, high)
    as_found = [
        CalibrationPoint(
            reference,
            specification.offset
            + specification.gain * 1.002 * reference
            + (index - 1) * specification.quantization,
        )
        for index, reference in enumerate(references)
    ]
    as_left = [
        CalibrationPoint(
            reference,
            specification.offset + specification.gain * reference,
        )
        for reference in references
    ]
    return as_found, as_left


def _campaign_phases(config: Mapping[str, Any]) -> list[CampaignPhase]:
    process = config["process"]
    phases: list[CampaignPhase] = []
    pattern = process["cip_pattern"]
    for cycle in range(1, int(process["process_cycles"]) + 1):
        kind = pattern[(cycle - 1) % len(pattern)]
        phases.extend(
            [
                CampaignPhase(
                    f"C{cycle:03d}-production",
                    "normal",
                    float(process["production_duration_s"]),
                ),
                CampaignPhase(
                    f"C{cycle:03d}-cip-{kind}",
                    "cip_cycle" if kind == "complete" else "incomplete_cleaning",
                    float(process["cip_duration_s"]),
                ),
                CampaignPhase(
                    f"C{cycle:03d}-restart",
                    "normal",
                    float(process["restart_duration_s"]),
                ),
            ]
        )
    return phases


def _time_at(start: datetime, elapsed_s: float) -> datetime:
    return start + timedelta(seconds=elapsed_s)


def _schema_table(fields: Sequence[str], role_overrides: Mapping[str, str]) -> dict[str, object]:
    return {
        "field_count": len(fields),
        "fields": [
            {"name": name, "role": role_overrides.get(name, "context")}
            for name in fields
        ],
    }


def _process_roles(fields: Sequence[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for name in fields:
        if name in {
            "campaign_id",
            "phase_id",
            "phase_index",
            "phase_kind",
            "phase_run_id",
            "campaign_time_start_s",
            "campaign_time_s",
            "scenario",
            "plant_mode",
            "time_start_s",
            "time_s",
            "step_dt_s",
            "run_id",
            "config_hash",
            "model_version",
            "random_seed",
        }:
            roles[name] = "context"
        elif name.startswith("alarm_") or name.endswith("_alarm_count"):
            roles[name] = "observable_alarm"
        elif name in {
            "control_temp_sensor_c",
            "safety_temp_sensor_c",
            "preheat_temp_sensor_c",
            "product_temp_sensor_c",
            "measured_flow_l_h",
            "measured_differential_pressure_bar",
            "fdv_position_feedback",
            "cip_conductivity_proxy_ms_cm",
            "cip_ph_proxy",
            "restart_product_interface_signal_fraction",
            "power_good_signal",
            "temperature_sensor_quality_ok",
        }:
            roles[name] = "observable_signal"
        elif name in {
            "steam_valve",
            "fdv_command_forward",
            "fdv_command_position",
            "cip_cycle_active",
            "cip_chemical_concentration_pct",
        }:
            roles[name] = "control"
        elif name.endswith("_l") or name in {"quality_out_of_spec_l"}:
            roles[name] = "outcome"
        else:
            roles[name] = "oracle"
    return roles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the integrated non-biological HTST reference digital twin."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run_integrated(config_path: Path, output_path: Path) -> Path:
    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
    config_hash = _sha256_json(config)
    digital_run_id = f"{config['run_id']}-{config_hash[:12]}"
    seed = int(config["seed"])

    plant = load_reference_plant()
    plant_validation = validate_reference_plant(plant)
    haccp_plan = load_haccp_plan()
    if haccp_plan["plan_id"] != config["haccp"]["plan_id"]:
        raise DigitalTwinConfigError("configured HACCP plan_id does not match haccp_plan.json")
    policy_file = str(config["lifecycle"]["policy_file"])
    policy_path = (config_path.parent / policy_file).resolve()
    try:
        policy_path.relative_to(config_path.parent)
    except ValueError as exc:
        raise DigitalTwinConfigError("lifecycle policy escapes config directory") from exc
    policy = load_maintenance_policy(policy_path)

    phases = _campaign_phases(config)
    process_duration_s = math.fsum(phase.duration_s for phase in phases)
    process_config = HTSTConfig(
        dt_s=float(config["process"]["dt_s"]),
        duration_s=process_duration_s,
        random_seed=seed,
        initial_fouling_index=float(config["process"]["initial_fouling_index"]),
    )
    simulator = HTSTSimulator(process_config)
    process_rows = simulator.run_campaign(
        phases,
        campaign_id=digital_run_id,
        reset=True,
    )
    if not process_rows:
        raise RuntimeError("integrated campaign emitted no rows")

    catalog = load_sensor_catalog()
    missing_tags = set(_SENSOR_TRUTH_FIELDS) - {
        specification.tag for specification in catalog.sensors
    }
    if missing_tags:
        raise RuntimeError(f"sensor catalog lacks integrated tags: {sorted(missing_tags)}")
    evaluation_start = datetime.fromisoformat(
        str(config["calibration"]["evaluation_date"])
    ).replace(tzinfo=timezone.utc)
    calibrated_at = evaluation_start - timedelta(days=1)
    records: dict[str, Any] = {}
    chains: dict[str, SensorChain] = {}
    calibration_summary: list[dict[str, object]] = []
    for index, tag in enumerate(sorted(_SENSOR_TRUTH_FIELDS)):
        specification = catalog.sensor(tag)
        found, left = _calibration_points(specification)
        record = create_calibration_record(
            specification,
            record_id=f"CAL-{tag}-SYNTHETIC-001",
            calibrated_at=calibrated_at,
            as_found_points=found,
            as_left_points=left,
            coverage_factor=float(config["calibration"]["coverage_factor"]),
        )
        records[tag] = record
        chains[tag] = SensorChain(specification, seed=seed + 10_000 + index)
        base_combined = math.sqrt(
            math.fsum(value * value for value in dict(record.uncertainty_components).values())
        )
        calibration_summary.append(
            {
                "sensor_tag": tag,
                "record_id": record.record_id,
                "calibrated_at": record.calibrated_at,
                "due_at": record.due_at,
                "as_found_intercept": record.as_found_fit.intercept,
                "as_found_slope": record.as_found_fit.slope,
                "as_left_intercept": record.as_left_fit.intercept,
                "as_left_slope": record.as_left_fit.slope,
                "combined_base_standard_uncertainty": base_combined,
                "coverage_factor": record.coverage_factor,
                "expanded_base_uncertainty": record.coverage_factor * base_combined,
                "status_at_run_start": record.status_at(evaluation_start),
                "claim_boundary": record.claim_boundary,
            }
        )

    plc = ReferencePLC(
        PLCSettings(
            scan_time_s=float(config["plc"]["scan_time_s"]),
            forward_confirmation_s=float(config["plc"]["forward_confirmation_s"]),
        )
    )
    ledger = EvidenceLedger(haccp_plan)
    lot_id = str(config["haccp"]["lot_id"])
    sensor_trace: list[dict[str, object]] = []
    plc_trace: list[dict[str, object]] = []
    plc_events: list[dict[str, object]] = []
    timeseries: list[dict[str, object]] = []
    previous_causes = ""

    for row in process_rows:
        campaign_time_s = float(row["campaign_time_s"])
        observed_at = _time_at(evaluation_start, campaign_time_s)
        corrected_row = dict(row)
        calibration_statuses: dict[str, object] = {}
        corrected_by_tag: dict[str, float | None] = {}
        valid_count = 0
        unknown_count = 0

        for tag in sorted(_SENSOR_TRUTH_FIELDS):
            specification = catalog.sensor(tag)
            source_field = _SENSOR_TRUTH_FIELDS[tag]
            truth = float(row[source_field])
            force_dropout = (
                not bool(row.get("sensor_available", 1)) if tag == "TT-104" else None
            )
            reading = chains[tag].step(
                truth,
                float(row["step_dt_s"]),
                force_dropout=force_dropout,
            )
            decision = evaluate_guarded_reading(
                specification,
                reading,
                records[tag],
                observed_at=observed_at,
            )
            correction = decision.correction
            corrected = (
                None
                if correction is None or correction.status != "VALID"
                else correction.corrected
            )
            corrected_by_tag[tag] = corrected
            calibration_valid = corrected is not None and reading.quality == "GOOD"
            calibration_statuses[tag] = {
                "status": "VALID" if calibration_valid else "UNKNOWN",
                "valid_until_time_s": process_duration_s + 1.0,
            }
            valid_count += int(calibration_valid)
            unknown_count += int(not calibration_valid)
            sensor_trace.append(
                {
                    "digital_twin_run_id": digital_run_id,
                    "campaign_time_s": campaign_time_s,
                    "phase_id": row["phase_id"],
                    "sensor_tag": tag,
                    "source_field": source_field,
                    "true_value": reading.true_value,
                    "filtered_value": reading.filtered_value,
                    "indicated_value": reading.value,
                    "quality": reading.quality,
                    "dropout": int(reading.dropout),
                    "held_last": int(reading.held_last),
                    "calibration_record_id": records[tag].record_id,
                    "correction_status": (
                        "UNAVAILABLE" if correction is None else correction.status
                    ),
                    "corrected_value": corrected,
                    "combined_standard_uncertainty": (
                        None if correction is None else correction.combined_standard_uncertainty
                    ),
                    "expanded_uncertainty": (
                        None if correction is None else correction.expanded_uncertainty
                    ),
                    "lower_bound": None if correction is None else correction.lower_bound,
                    "upper_bound": None if correction is None else correction.upper_bound,
                    "guard_status": decision.status,
                    "guard_disposition": decision.disposition,
                }
            )

        for tag, process_field in _CORRECTED_PROCESS_FIELDS.items():
            corrected = corrected_by_tag[tag]
            if corrected is not None:
                corrected_row[process_field] = corrected
        if corrected_by_tag["TT-104"] is None:
            corrected_row["temperature_sensor_quality_ok"] = 0
        if corrected_by_tag["FT-101"] is None:
            corrected_row["measured_flow_l_h"] = float(row["maximum_safe_flow_l_h"]) + 1.0
        corrected_flow = float(corrected_row["measured_flow_l_h"])
        if corrected_flow > 0.0:
            corrected_row["estimated_fastest_residence_time_s"] = (
                process_config.holding_tube_volume_l
                / (corrected_flow / 3600.0)
                * process_config.fastest_flow_efficiency
            )
        else:
            corrected_row["estimated_fastest_residence_time_s"] = 0.0
        if corrected_by_tag["PDT-101"] is None:
            corrected_row["measured_differential_pressure_bar"] = -0.5
        corrected_row["sensor_disagreement_c"] = abs(
            float(corrected_row["control_temp_sensor_c"])
            - float(corrected_row["safety_temp_sensor_c"])
        )

        plc_result = plc.scan(corrected_row)
        output_image = plc_result["output_image"]
        plc_row = {
            "digital_twin_run_id": digital_run_id,
            "campaign_time_s": campaign_time_s,
            "phase_id": row["phase_id"],
            "scan_index": plc_result["scan_index"],
            "previous_state": plc_result["previous_state"],
            "plc_state": plc_result["plc_state"],
            "state_changed": plc_result["state_changed"],
            "cip_recipe_state": plc_result["cip_recipe_state"],
            "forward_permissive": plc_result["forward_permissive"],
            "trip_latched": plc_result["trip_latched"],
            "active_causes": plc_result["active_causes"],
            "active_trip_causes": plc_result["active_trip_causes"],
            "active_divert_causes": plc_result["active_divert_causes"],
            "feed_pump_command": int(output_image["feed_pump_command"]),
            "booster_pump_command": int(output_image["booster_pump_command"]),
            "steam_enable": int(output_image["steam_enable"]),
            "fdv_command_forward": int(output_image["fdv_command_forward"]),
            "fdv_command_divert": int(output_image["fdv_command_divert"]),
            "cip_pump_command": int(output_image["cip_pump_command"]),
            "cip_rinse_valve": int(output_image["cip_rinse_valve"]),
            "cip_alkali_valve": int(output_image["cip_alkali_valve"]),
            "cip_acid_valve": int(output_image["cip_acid_valve"]),
            "cip_drain_valve": int(output_image["cip_drain_valve"]),
            "core_fdv_command_forward": int(row["fdv_command_forward"]),
            "shadow_core_fdv_command_match": int(
                bool(output_image["fdv_command_forward"])
                == bool(row["fdv_command_forward"])
            ),
        }
        plc_trace.append(plc_row)
        causes = str(plc_result["active_causes"])
        if plc_result["state_changed"] or causes != previous_causes:
            plc_events.append(
                {
                    "digital_twin_run_id": digital_run_id,
                    "campaign_time_s": campaign_time_s,
                    "phase_id": row["phase_id"],
                    "event_type": "STATE_CHANGE" if plc_result["state_changed"] else "CAUSE_CHANGE",
                    "previous_state": plc_result["previous_state"],
                    "plc_state": plc_result["plc_state"],
                    "active_causes": causes,
                }
            )
        previous_causes = causes

        haccp_result = ledger.evaluate(
            corrected_row,
            calibration_statuses,
            lot_id=lot_id,
        )
        timeseries.append(
            {
                "digital_twin_run_id": digital_run_id,
                "campaign_time_start_s": row["campaign_time_start_s"],
                "campaign_time_s": campaign_time_s,
                "phase_id": row["phase_id"],
                "phase_kind": row["phase_kind"],
                "scenario": row["scenario"],
                "plant_mode": row["plant_mode"],
                "control_temp_sensor_c": row["control_temp_sensor_c"],
                "safety_temp_sensor_c": row["safety_temp_sensor_c"],
                "calibrated_safety_temp_c": corrected_by_tag["TT-104"],
                "measured_flow_l_h": row["measured_flow_l_h"],
                "calibrated_flow_l_h": corrected_by_tag["FT-101"],
                "estimated_fastest_residence_time_s": row[
                    "estimated_fastest_residence_time_s"
                ],
                "calibrated_fastest_residence_time_s": corrected_row[
                    "estimated_fastest_residence_time_s"
                ],
                "measured_differential_pressure_bar": row[
                    "measured_differential_pressure_bar"
                ],
                "calibrated_differential_pressure_bar": corrected_by_tag["PDT-101"],
                "fdv_command_forward": row["fdv_command_forward"],
                "fdv_actual_forward": row["fdv_actual_forward"],
                "plc_state": plc_result["plc_state"],
                "plc_fdv_command_forward": int(output_image["fdv_command_forward"]),
                "plc_active_causes": causes,
                "calibration_valid_count": valid_count,
                "calibration_unknown_count": unknown_count,
                "haccp_overall_status": haccp_result["overall_status"],
                "haccp_lot_status": haccp_result["lot_status"],
                "product_boundary_l": row["product_boundary_l"],
                "unsafe_forward_l": row["unsafe_forward_l"],
                "quality_out_of_spec_l": row["quality_out_of_spec_l"],
                "fouling_index": row["fouling_index"],
                "cip_total_soil_g": row["cip_total_soil_g"],
                "cip_release_permissive": row["cip_release_permissive"],
            }
        )

    lifecycle_result = simulate_lifecycle(
        policy,
        cycles=int(config["lifecycle"]["cycles"]),
        seed=seed,
    )
    haccp_export = ledger.export_json_safe()
    haccp_deviations = [
        {
            "record_id": record["record_id"],
            "sequence": record["sequence"],
            "time_s": record["time_s"],
            "lot_id": record["lot_id"],
            "record_type": record["record_type"],
            "control_id": record.get("control_id", ""),
            "limit_id": record.get("limit_id", ""),
            "reason": record.get("reason", ""),
            "lot_status_after": record.get("lot_status_after", ""),
            "record_hash": record["record_hash"],
        }
        for record in ledger.records
        if record["record_type"] in {"DEVIATION_RECORD", "CONTAINMENT_ACTION"}
    ]
    if not haccp_deviations:
        haccp_deviations = []

    haccp_status_counts = Counter(row["haccp_overall_status"] for row in timeseries)
    phase_counts = Counter(str(row["phase_kind"]) for row in process_rows)
    summary = {
        "digital_twin_version": DIGITAL_TWIN_VERSION,
        "core_model_version": MODEL_VERSION,
        "model_status": MODEL_STATUS,
        "digital_twin_run_id": digital_run_id,
        "config_hash": config_hash,
        "biological_model": {
            "implemented": False,
            "reason": (
                "Organism-specific biology, CFU, D-value, growth and challenge-study "
                "models are explicitly excluded by user decision."
            ),
            "legacy_core_note": (
                "The v2.2 core retains a dimensionless relative heat-treatment "
                "diagnostic; v3 HACCP, PLC, calibration and lifecycle decisions do "
                "not treat it as biological evidence."
            ),
        },
        "reference_plant": plant_validation,
        "process": {
            "rows": len(process_rows),
            "duration_s": float(process_rows[-1]["campaign_time_s"]),
            "phase_counts": dict(sorted(phase_counts.items())),
            "forward_l": math.fsum(float(row["forward_l"]) for row in process_rows),
            "unsafe_forward_l": math.fsum(
                float(row["unsafe_forward_l"]) for row in process_rows
            ),
            "quality_out_of_spec_l": math.fsum(
                float(row["quality_out_of_spec_l"]) for row in process_rows
            ),
            "transition_drained_l": math.fsum(
                float(row["transition_drained_l"]) for row in process_rows
            ),
            "final_fouling_index": float(process_rows[-1]["fouling_index"]),
            "final_surface_soil_g": float(process_rows[-1]["surface_total_soil_g"]),
        },
        "instrumentation": {
            "sensor_count": len(records),
            "sensor_trace_rows": len(sensor_trace),
            "calibration_records": len(records),
            "unknown_readings": sum(
                int(row["correction_status"] != "VALID") for row in sensor_trace
            ),
            "guard_failures": sum(
                int(row["guard_status"] == "FAIL") for row in sensor_trace
            ),
            "status": "synthetic_reference_no_traceability_claim",
        },
        "plc_shadow": {
            "scan_count": len(plc_trace),
            "event_count": len(plc_events),
            "state_counts": dict(
                sorted(Counter(row["plc_state"] for row in plc_trace).items())
            ),
            "core_command_match_scans": sum(
                int(row["shadow_core_fdv_command_match"]) for row in plc_trace
            ),
            "shadow_only": True,
        },
        "haccp_evidence": {
            "record_count": haccp_export["record_count"],
            "chain_head": haccp_export["chain_head"],
            "chain_valid": haccp_export["chain_verification"]["valid"],
            "status_counts": dict(sorted(haccp_status_counts.items())),
            "lot_statuses": haccp_export["lot_statuses"],
            "conformity_status": "not_assessed",
            "automatic_product_release": False,
        },
        "lifecycle": lifecycle_result.summary,
        "limitations": [
            "The P&ID is a public-reference topology, not an as-built drawing.",
            "The PLC is shadow-only and does not actuate the governing simulator or equipment.",
            "Calibration records and uncertainty budgets are synthetic and not traceable certificates.",
            "HACCP conformity and product release are not assessed or automated.",
            "Lifecycle coefficients are not fitted to plant maintenance history.",
            "No organism-specific biology, CFU, D-value, growth or challenge-study model is included; the v2.2 core's dimensionless relative heat-treatment diagnostic is not biological evidence.",
        ],
    }

    process_fields = tuple(process_rows[0])
    core_events = extract_events(process_rows)
    calibration_records_payload = {
        "schema_version": "1.0.0",
        "status": "synthetic_reference",
        "records": [records[tag].to_dict() for tag in sorted(records)],
    }
    reference_markdown = (
        "# Reference P&ID\n\n"
        "> Unvalidated public-reference topology; not an as-built drawing.\n\n"
        "```mermaid\n"
        + render_mermaid(plant)
        + "\n```\n"
    )
    schema = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "digital_twin_version": DIGITAL_TWIN_VERSION,
        "roles": {
            "context": "Identifiers, clocks, versions and phase information.",
            "observable_signal": "Synthetic instrumentation boundary signal.",
            "observable_alarm": "Alarm derived from observable/status inputs.",
            "control": "Command or controller state.",
            "oracle": "Privileged physical truth or internal state; not an ML feature.",
            "outcome": "Routing, evidence, quality or lifecycle outcome.",
        },
        "tables": {
            "process_trace.csv": _schema_table(
                process_fields, _process_roles(process_fields)
            ),
            "timeseries.csv": _schema_table(
                TIMESERIES_FIELDS,
                {
                    name: (
                        "outcome"
                        if name.startswith("haccp_")
                        or name.endswith("_l")
                        or name in {"fouling_index", "cip_total_soil_g"}
                        else "control"
                        if name.startswith("plc_") or name.startswith("fdv_")
                        else "observable_signal"
                        if "sensor" in name
                        or "measured" in name
                        or "calibrated" in name
                        or "residence" in name
                        else "context"
                    )
                    for name in TIMESERIES_FIELDS
                },
            ),
            "sensor_trace.csv": _schema_table(
                SENSOR_TRACE_FIELDS,
                {
                    name: (
                        "oracle"
                        if name in {"true_value", "filtered_value"}
                        else "outcome"
                        if name.startswith("guard_") or name.startswith("correction_")
                        else "observable_signal"
                        if name
                        in {
                            "indicated_value",
                            "quality",
                            "corrected_value",
                            "combined_standard_uncertainty",
                            "expanded_uncertainty",
                            "lower_bound",
                            "upper_bound",
                        }
                        else "context"
                    )
                    for name in SENSOR_TRACE_FIELDS
                },
            ),
            "plc_trace.csv": _schema_table(
                PLC_TRACE_FIELDS,
                {name: "control" if "command" in name or name == "plc_state" else "context" for name in PLC_TRACE_FIELDS},
            ),
            "lifecycle_trace.csv": _schema_table(
                LIFECYCLE_TRACE_FIELDS,
                {name: "outcome" if "rul" in name or name in {"irreversible_damage", "reversible_fouling"} else "context" for name in LIFECYCLE_TRACE_FIELDS},
            ),
            "service_intervals.csv": _schema_table(
                SERVICE_INTERVAL_FIELDS,
                {name: "outcome" if name in {"event_observed", "exact_rul_h", "rul_lower_bound_h", "failure_cause"} else "context" for name in SERVICE_INTERVAL_FIELDS},
            ),
            "haccp_evidence.jsonl": {
                "record_count": haccp_export["record_count"],
                "role": "outcome",
                "integrity": "canonical SHA-256 append-only chain",
            },
        },
    }

    source = source_provenance(
        {
            "model.py": HERE / "model.py",
            "process_components.py": HERE / "process_components.py",
            "reference_pid.json": HERE / "reference_pid.json",
            "reference_plant.py": HERE / "reference_plant.py",
            "plc_logic.py": HERE / "plc_logic.py",
            "plc/cause_effect.json": HERE / "plc" / "cause_effect.json",
            "plc/htst_reference.st": HERE / "plc" / "htst_reference.st",
            "sensor_catalog.json": HERE / "sensor_catalog.json",
            "sensor_calibration.py": HERE / "sensor_calibration.py",
            "haccp_plan.json": HERE / "haccp_plan.json",
            "haccp.py": HERE / "haccp.py",
            "maintenance_policy.json": policy_path,
            "lifecycle.py": HERE / "lifecycle.py",
            "run_digital_twin.py": Path(__file__).resolve(),
        }
    )

    output = output_path.expanduser().resolve()
    artifact_names = [
        "config_snapshot.json",
        "reference_plant_snapshot.json",
        "reference_plant.md",
        "process_trace.csv",
        "process_events.csv",
        "timeseries.csv",
        "sensor_trace.csv",
        "calibration_records.json",
        "calibration_summary.csv",
        "plc_trace.csv",
        "plc_events.csv",
        "haccp_evidence.jsonl",
        "haccp_deviations.csv",
        "lifecycle_trace.csv",
        "lifecycle_events.csv",
        "service_intervals.csv",
        "digital_twin_summary.json",
        "schema.json",
        "run_manifest.json",
    ]
    with staged_output_directory(output) as staging:
        _write_json(staging / "config_snapshot.json", config)
        _write_json(staging / "reference_plant_snapshot.json", plant)
        (staging / "reference_plant.md").write_text(reference_markdown, encoding="utf-8")
        _write_csv(staging / "process_trace.csv", process_rows, process_fields)
        _write_csv(staging / "process_events.csv", core_events)
        _write_csv(staging / "timeseries.csv", timeseries, TIMESERIES_FIELDS)
        _write_csv(staging / "sensor_trace.csv", sensor_trace, SENSOR_TRACE_FIELDS)
        _write_json(staging / "calibration_records.json", calibration_records_payload)
        _write_csv(
            staging / "calibration_summary.csv",
            calibration_summary,
            CALIBRATION_SUMMARY_FIELDS,
        )
        _write_csv(staging / "plc_trace.csv", plc_trace, PLC_TRACE_FIELDS)
        _write_csv(staging / "plc_events.csv", plc_events, PLC_EVENT_FIELDS)
        _write_jsonl(staging / "haccp_evidence.jsonl", ledger.records)
        _write_csv(
            staging / "haccp_deviations.csv",
            haccp_deviations,
            HACCP_DEVIATION_FIELDS,
        )
        _write_csv(
            staging / "lifecycle_trace.csv",
            lifecycle_result.trace,
            LIFECYCLE_TRACE_FIELDS,
        )
        _write_csv(
            staging / "lifecycle_events.csv",
            lifecycle_result.events,
            LIFECYCLE_EVENT_FIELDS,
        )
        _write_csv(
            staging / "service_intervals.csv",
            lifecycle_result.service_intervals,
            SERVICE_INTERVAL_FIELDS,
        )
        _write_json(staging / "digital_twin_summary.json", summary)
        _write_json(staging / "schema.json", schema)
        manifest = {
            "digital_twin_version": DIGITAL_TWIN_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "core_model_version": MODEL_VERSION,
            "model_status": MODEL_STATUS,
            "digital_twin_run_id": digital_run_id,
            "config_hash": config_hash,
            "seed": seed,
            "python_version": platform.python_version(),
            "components": {
                "reference_pid": plant["schema_version"],
                "plc_shadow": "1.0.0",
                "sensor_calibration": "1.0.0",
                "haccp_evidence": "1.0.0",
                "lifecycle": lifecycle_result.summary["lifecycle_model_version"],
                "biological_model": "excluded",
            },
            "source_provenance": source,
            "artifacts": artifact_names,
            "reproduce": (
                "python3 run_digital_twin.py "
                "--config digital_twin_config.json "
                "--output <output_dir>"
            ),
            "counts": {
                "process_rows": len(process_rows),
                "sensor_rows": len(sensor_trace),
                "plc_rows": len(plc_trace),
                "haccp_records": len(ledger.records),
                "lifecycle_trace_rows": len(lifecycle_result.trace),
            },
        }
        _write_json(staging / "run_manifest.json", manifest)
        write_checksums(
            staging,
            [staging / name for name in artifact_names],
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = run_integrated(args.config, args.output)
    except (DigitalTwinConfigError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"digital twin run failed: {exc}") from exc
    summary = json.loads((output / "digital_twin_summary.json").read_text(encoding="utf-8"))
    print(
        "Integrated HTST reference run complete: "
        f"rows={summary['process']['rows']}, "
        f"HACCP records={summary['haccp_evidence']['record_count']}, "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
