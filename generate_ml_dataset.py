#!/usr/bin/env python3
"""Generate leakage-safe, counterfactual HTST episodes for ML experiments."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml_pipeline_common import (
    BASE_DIR,
    CONTRACT_PATH,
    feature_fields,
    field_type,
    load_contract,
    select_taxonomy,
    sha256_file,
    sha256_json,
    stable_seed,
    write_checksums,
    write_csv_header,
    write_json,
)
from model import HTSTConfig, HTSTSimulator, MODEL_VERSION, SCENARIO_NAMES


GENERATOR_VERSION = "2.2.1"
DEFAULT_OUTPUT = BASE_DIR / "ml_datasets" / "D1-pilot"
DEFAULT_OOD_CONTRACT = BASE_DIR / "ood_profile_contract.json"
SEMANTIC_VERSION = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
DOMAIN_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")

EXCLUDED_PRECOMPUTED_DIAGNOSTICS = {
    "alarm_count",
    "observable_alarm_count",
    "diagnostic_alarm_count",
    "oracle_alarm_count",
    "total_alarm_count",
    "alarm_unsafe_forward",
    "bounded_exponential_count",
    "bounded_exponential_diagnostic",
    "diagnostic_count",
}

ORACLE_SOURCE_FIELDS = [
    "alarm_count",
    "observable_alarm_count",
    "diagnostic_alarm_count",
    "oracle_alarm_count",
    "total_alarm_count",
    "alarm_unsafe_forward",
    "bounded_exponential_count",
    "bounded_exponential_diagnostic",
    "diagnostic_count",
    "flow_l_h",
    "inlet_temp_c",
    "preheat_temp_c",
    "heater_out_temp_c",
    "fresh_feed_l",
    "return_to_balance_tank_l",
    "return_line_inventory_l",
    "balance_tank_volume_l",
    "balance_tank_temp_c",
    "balance_tank_mean_pass_count",
    "balance_tank_risk_fraction",
    "balance_tank_chemical_fraction",
    "balance_tank_product_fraction",
    "inlet_mean_pass_count",
    "inlet_recycle_risk_fraction",
    "inlet_chemical_fraction",
    "inlet_product_fraction",
    "holding_chemical_fraction",
    "holding_product_fraction",
    "shadow_regenerator_cold_out_c",
    "shadow_regenerator_hot_out_c",
    "shadow_regenerator_wall_temp_c",
    "shadow_regenerator_effective_ua_kw_k",
    "shadow_regenerator_energy_balance_error_kj",
    "shadow_heater_product_out_c",
    "shadow_heater_utility_out_c",
    "shadow_heater_wall_temp_c",
    "shadow_heater_effective_ua_kw_k",
    "shadow_heater_energy_balance_error_kj",
    "actual_safe",
    "actual_residence_time_s",
    "mean_transport_residence_time_s",
    "fastest_residence_time_s",
    "fastest_flow_efficiency",
    "holding_out_temp_c",
    "relative_lethality",
    "outlet_thermal_safe_fraction",
    "routed_temp_c",
    "routed_residence_time_s",
    "routed_fastest_residence_time_s",
    "routed_transport_residence_time_s",
    "routed_relative_lethality",
    "routed_thermal_safe_fraction",
    "routed_pressure_safe_fraction",
    "routed_contamination_risk_fraction",
    "routed_chemical_fraction",
    "routed_product_fraction",
    "routed_hygiene_risk_fraction",
    "forward_temp_c",
    "forward_residence_time_s",
    "forward_fastest_residence_time_s",
    "forward_transport_residence_time_s",
    "forward_relative_lethality",
    "forward_thermal_safe_fraction",
    "forward_min_process_differential_pressure_bar",
    "forward_pressure_safe_fraction",
    "forward_contamination_risk_fraction",
    "forward_chemical_fraction",
    "forward_product_fraction",
    "forward_hygiene_risk_fraction",
    "differential_pressure_bar",
    "raw_pressure_bar",
    "pasteurized_pressure_bar",
    "booster_pump_factor",
    "leak_fraction",
    "contamination_risk",
    "fouling_index",
    "surface_protein_soil_g",
    "surface_mineral_soil_g",
    "surface_total_soil_g",
    "surface_hygiene_risk_fraction",
    "fdv_actual_forward",
    "fdv_position",
    "fdv_forward_fraction",
    "fdv_mismatch_time_s",
    "routed_volume_l",
    "valve_forward_l",
    "post_fdv_inlet_l",
    "product_boundary_l",
    "safe_forward_l",
    "unsafe_forward_l",
    "thermal_unsafe_forward_l",
    "pressure_noncompliant_forward_l",
    "contamination_exposed_forward_l",
    "chemical_noncompliant_forward_l",
    "dilution_noncompliant_forward_l",
    "hygiene_noncompliant_forward_l",
    "transition_drained_l",
    "transition_added_l",
    "quality_out_of_spec_l",
    "forward_l",
    "diverted_l",
    "cip_recirculated_l",
    "product_temp_c",
    "potential_product_temp_c",
    "product_downstream_fault_fraction",
    "product_regenerator_factor",
    "product_cooler_factor",
    "shadow_cooler_product_out_c",
    "shadow_cooler_utility_out_c",
    "shadow_cooler_wall_temp_c",
    "shadow_cooler_effective_ua_kw_k",
    "shadow_cooler_energy_balance_error_kj",
    "return_temp_c",
    "diverted_mean_pass_count",
    "diverted_recycle_risk_fraction",
    "diverted_chemical_fraction",
    "diverted_product_fraction",
    "heat_kw",
    "regeneration_saved_kw",
    "pump_power_kw",
    "heat_energy_kwh",
    "pump_energy_kwh",
    "holding_inventory_l",
    "fdv_line_inventory_l",
    "post_fdv_inventory_l",
    "mass_balance_error_l",
    "post_fdv_balance_error_l",
    "balance_tank_volume_error_l",
    "balance_tank_temperature_moment_error_l_c",
    "balance_tank_pass_moment_error_l",
    "balance_tank_risk_volume_error_l",
    "balance_tank_chemical_volume_error_l",
    "balance_tank_product_volume_error_l",
    "balance_tank_ambient_heat_kj",
    "cip_effective_chemical_concentration_pct",
    "cip_protein_soil_g",
    "cip_mineral_soil_g",
    "cip_total_soil_g",
    "cip_removed_protein_g",
    "cip_removed_mineral_g",
    "cip_residual_chemical_fraction",
    "cip_cleaning_complete",
    "control_sensor_bias_c",
    "safety_sensor_bias_c",
    "flow_meter_bias_fraction",
    "heater_capacity_factor",
    "regenerator_factor",
    "cooler_factor",
    "power_available",
    "sensor_available",
    "valve_travel_time_factor",
    "valve_leakage_fraction",
    "cleaning_effectiveness_factor",
]

DERIVED_LABEL_FIELDS = [
    "safety_event",
    "unsafe_forward_event",
    "diversion_active",
    "diversion_event",
    "cooling_excursion",
    "mode_transition_event",
]

LABEL_ID_FIELDS = [
    "episode_id",
    "time_start_s",
    "time_s",
    "canonical_code",
    "canonical_scenario",
    "simulator_scenario",
    "fault_active",
    "fault_injection_time_s",
    "fault_end_time_s",
    "fault_effect_time_s",
    "behavior_active",
    "behavior_injection_time_s",
    "behavior_end_time_s",
    "behavior_effect_time_s",
    "physical_effect_time_s",
    "detection_eligible",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-version", default="D1-pilot")
    parser.add_argument("--profiles", type=int, default=12)
    parser.add_argument(
        "--ood-profiles",
        type=int,
        default=0,
        help="Additional synthetic support-shift profiles held out from ID fitting",
    )
    parser.add_argument(
        "--ood-contract",
        type=Path,
        default=DEFAULT_OOD_CONTRACT,
        help="Versioned synthetic OOD profile-domain contract",
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--dt-s", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Canonical codes/names or aliases (comma-separated values also accepted)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.profiles < 1 or args.replicates < 1:
        raise ValueError("profiles and replicates must be positive")
    if args.ood_profiles < 0:
        raise ValueError("ood-profiles must be nonnegative")
    if args.duration_s <= 0.0:
        raise ValueError("duration-s must be positive")
    if not 0.0 < args.dt_s <= 1.0:
        raise ValueError("dt-s must be in (0, 1]")
    if args.duration_s < 20.0:
        raise ValueError("duration-s must be at least 20 seconds")


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Read a finite, duplicate-key-free JSON object."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read OOD profile contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("OOD profile contract must be a JSON object")
    return value


def _numeric_range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element JSON array")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError(f"{label} bounds must be real numbers")
    lower, upper = map(float, value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"{label} must contain finite increasing bounds")
    return lower, upper


def _ranges_have_strict_gap(
    left: Mapping[str, tuple[float, float]],
    right: Mapping[str, tuple[float, float]],
) -> bool:
    return any(
        left[name][1] < right[name][0] or right[name][1] < left[name][0]
        for name in left
    )


def _validate_physical_range_box(
    domain: str, ranges: Mapping[str, tuple[float, float]]
) -> None:
    """Fail closed on simulator constraints for the complete parameter box.

    HTSTConfig has two joint constraints among sampled profile fields.  We
    prove those at their adverse corners, then exercise every scalar endpoint
    against the box midpoint so model-level bounds remain the source of truth.
    """

    if (
        ranges["balance_tank_initial_volume_l"][1]
        > ranges["balance_tank_capacity_l"][0]
    ):
        raise ValueError(
            f"{domain} permits balance_tank_initial_volume_l above capacity"
        )
    if (
        72.0 + ranges["forward_temperature_margin_c"][1]
        >= ranges["pasteurization_setpoint_c"][0]
    ):
        raise ValueError(
            f"{domain} permits diversion threshold plus margin at/above setpoint"
        )
    if ranges["nominal_holding_time_s"][0] < 15.0:
        raise ValueError(
            f"{domain} permits nominal holding time below the model minimum"
        )

    midpoint = {
        name: (bounds[0] + bounds[1]) / 2.0 for name, bounds in ranges.items()
    }
    candidates = [
        {name: bounds[0] for name, bounds in ranges.items()},
        {name: bounds[1] for name, bounds in ranges.items()},
        midpoint,
    ]
    for name, bounds in ranges.items():
        for endpoint in bounds:
            candidate = dict(midpoint)
            candidate[name] = endpoint
            candidates.append(candidate)
    for candidate in candidates:
        try:
            HTSTConfig(**candidate, duration_s=20.0, dt_s=1.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{domain} contains an HTSTConfig-invalid profile corner: {exc}"
            ) from exc


def _load_ood_profile_contract(
    path: Path, id_ranges_raw: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, tuple[float, float]]]]:
    """Validate and expand the synthetic OOD contract into complete range boxes."""

    contract = _strict_json_object(path)
    required = {
        "version",
        "id_domain",
        "ood_domains",
        "generation_policy",
        "profile_parameter_range_overrides",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"OOD profile contract is missing keys: {missing}")
    if not SEMANTIC_VERSION.fullmatch(str(contract["version"])):
        raise ValueError("OOD profile contract version must be semantic")
    if contract["id_domain"] != "ID":
        raise ValueError("OOD profile contract id_domain must be exactly 'ID'")
    if not isinstance(contract["generation_policy"], dict) or not contract[
        "generation_policy"
    ]:
        raise ValueError("OOD profile contract generation_policy must be non-empty")

    names = contract["ood_domains"]
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not DOMAIN_NAME.fullmatch(name)
            for name in names
        )
        or len(names) != len(set(names))
        or "ID" in names
    ):
        raise ValueError("ood_domains must contain unique non-ID uppercase domain names")
    overrides = contract["profile_parameter_range_overrides"]
    if not isinstance(overrides, dict) or set(overrides) != set(names):
        raise ValueError(
            "profile_parameter_range_overrides must define exactly every OOD domain"
        )

    id_ranges = {
        name: _numeric_range(value, f"ID.{name}")
        for name, value in sorted(id_ranges_raw.items())
    }
    expanded: dict[str, dict[str, tuple[float, float]]] = {"ID": id_ranges}
    for domain in names:
        domain_overrides = overrides[domain]
        if not isinstance(domain_overrides, dict) or not domain_overrides:
            raise ValueError(f"{domain} must override at least one profile range")
        unknown = sorted(set(domain_overrides) - set(id_ranges))
        if unknown:
            raise ValueError(f"{domain} overrides unknown profile fields: {unknown}")
        complete = dict(id_ranges)
        for name, value in sorted(domain_overrides.items()):
            complete[name] = _numeric_range(value, f"{domain}.{name}")
        _validate_physical_range_box(domain, complete)
        expanded[domain] = complete
    _validate_physical_range_box("ID", id_ranges)

    domains = ["ID", *names]
    for left_index, left_name in enumerate(domains):
        for right_name in domains[left_index + 1 :]:
            if not _ranges_have_strict_gap(expanded[left_name], expanded[right_name]):
                raise ValueError(
                    f"Profile supports {left_name} and {right_name} are not strictly disjoint"
                )
    return contract, expanded


def _prepare_output(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use a new directory to preserve provenance."
            )
    else:
        path.mkdir(parents=True)


def _sample_profile(
    contract: Mapping[str, Any], master_seed: int, profile_index: int
) -> tuple[str, int, dict[str, float]]:
    seed = stable_seed(master_seed, "profile", profile_index)
    rng = random.Random(seed)
    sampled = {
        name: rng.uniform(float(bounds[0]), float(bounds[1]))
        for name, bounds in sorted(contract["profile_parameter_ranges"].items())
    }
    # Keep the forward permissive strictly below the setpoint after joint sampling.
    sampled["forward_temperature_margin_c"] = min(
        sampled["forward_temperature_margin_c"],
        sampled["pasteurization_setpoint_c"] - 72.0 - 0.05,
    )
    profile_hash = sha256_json(sampled)
    profile_id = f"P{profile_index:04d}-{profile_hash[:10]}"
    return profile_id, seed, sampled


def _sample_ood_profile(
    ranges: Mapping[str, tuple[float, float]],
    master_seed: int,
    ood_profile_index: int,
    global_profile_index: int,
    domain: str,
) -> tuple[str, int, dict[str, float]]:
    seed = stable_seed(master_seed, "ood-profile", domain, ood_profile_index)
    rng = random.Random(seed)
    sampled = {
        name: rng.uniform(float(bounds[0]), float(bounds[1]))
        for name, bounds in sorted(ranges.items())
    }
    # This cap is normally inactive for the validated contract, but keeps the
    # same fail-safe numerical guard used by legacy ID sampling.
    sampled["forward_temperature_margin_c"] = min(
        sampled["forward_temperature_margin_c"],
        sampled["pasteurization_setpoint_c"] - 72.0 - 0.05,
    )
    HTSTConfig(**sampled, duration_s=20.0, dt_s=1.0)
    profile_hash = sha256_json(sampled)
    profile_id = (
        f"P{global_profile_index:04d}-{domain}-{profile_hash[:10]}"
    )
    return profile_id, seed, sampled


def _sample_fault_schedule(
    contract: Mapping[str, Any], master_seed: int, profile_id: str, replicate: int,
    duration_s: float, dt_s: float,
) -> tuple[int, int, float, float, float]:
    sampling_seed = stable_seed(master_seed, "fault", profile_id, replicate)
    noise_seed = stable_seed(master_seed, "noise", profile_id, replicate)
    rng = random.Random(sampling_seed)
    onset_bounds = contract["fault_sampling"]["onset_fraction"]
    raw_onset = duration_s * rng.uniform(float(onset_bounds[0]), float(onset_bounds[1]))
    onset_s = min(duration_s - dt_s, max(dt_s, round(raw_onset / dt_s) * dt_s))
    duration_bounds = contract["fault_sampling"]["duration_s"]
    maximum_duration = min(float(duration_bounds[1]), duration_s - onset_s)
    minimum_duration = min(float(duration_bounds[0]), maximum_duration)
    fault_duration_s = rng.uniform(minimum_duration, maximum_duration)
    fault_duration_s = max(dt_s, round(fault_duration_s / dt_s) * dt_s)
    fault_duration_s = min(duration_s - onset_s, fault_duration_s)
    severity_bounds = contract["fault_sampling"]["severity"]
    severity = rng.uniform(float(severity_bounds[0]), float(severity_bounds[1]))
    return sampling_seed, noise_seed, onset_s, fault_duration_s, severity


def _sample_cip_fault_schedule(
    master_seed: int,
    profile_id: str,
    replicate: int,
    duration_s: float,
    dt_s: float,
) -> tuple[float, float]:
    """Place ineffective cleaning inside a chemically active CIP phase.

    The simulator's recipe starts caustic circulation at 60 s. A generic
    production-fault onset sampled at 25--75% of a long episode would often
    occur after the recipe and create a label with no physical intervention.
    """
    phase_start_s = 60.0
    phase_end_s = min(240.0, duration_s)
    if phase_end_s - phase_start_s < dt_s - 1e-12:
        raise ValueError(
            "F12 incomplete_cleaning requires duration-s to extend beyond "
            "the 60 s CIP pre-rinse"
        )
    minimum_active_s = min(15.0, phase_end_s - phase_start_s)
    latest_onset_s = max(
        phase_start_s,
        phase_end_s - minimum_active_s,
    )
    rng = random.Random(
        stable_seed(master_seed, "cip-fault", profile_id, replicate)
    )
    raw_onset = rng.uniform(phase_start_s, latest_onset_s)
    onset_s = round(raw_onset / dt_s) * dt_s
    onset_s = min(phase_end_s - dt_s, max(phase_start_s, onset_s))
    return onset_s, phase_end_s - onset_s


def _simulator_name(item: Mapping[str, Any]) -> str:
    """Always emit the canonical simulator implementation, never a legacy alias."""
    canonical = str(item["canonical_scenario"])
    scenarios = list(item.get("simulator_scenarios", []))
    if canonical in scenarios:
        return canonical
    if not scenarios:
        raise ValueError(f"No simulator scenario exists for {item['code']}")
    return str(scenarios[0])


def _first_observable_effect(
    normal_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
    observable_fields: Sequence[str],
    injection_time_s: float,
) -> float | None:
    if len(normal_rows) != len(scenario_rows):
        raise ValueError("Counterfactual episodes have different row counts")
    for normal, fault in zip(normal_rows, scenario_rows, strict=True):
        if normal["time_s"] != fault["time_s"]:
            raise ValueError("Counterfactual episodes have misaligned timestamps")
        if float(fault["time_s"]) + 1e-12 < injection_time_s:
            continue
        for field in observable_fields:
            left, right = normal[field], fault[field]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                tolerance = 1e-10 * max(1.0, abs(float(left)), abs(float(right)))
                if abs(float(left) - float(right)) > tolerance:
                    return float(fault["time_start_s"])
            elif left != right:
                return float(fault["time_start_s"])
    return None


def _event_onsets(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    onsets: list[float] = []
    previous = 0
    for row in rows:
        current = int(row[field])
        if current and not previous:
            onsets.append(float(row["time_s"]))
        previous = current
    return onsets


def _future_label(onsets: Sequence[float], time_s: float, horizon_s: float) -> int:
    index = bisect.bisect_right(onsets, time_s + 1e-12)
    return int(index < len(onsets) and onsets[index] <= time_s + horizon_s + 1e-12)


def _build_oracle_rows(
    rows: Sequence[Mapping[str, Any]],
    item: Mapping[str, Any],
    episode_id: str,
    simulator_scenario: str,
    behavior_start_s: float,
    behavior_duration_s: float,
    effect_time_s: float | None,
    physical_effect_time_s: float | None,
    detection_eligible: bool,
    horizons: Sequence[float],
    maximum_product_temp_c: float,
) -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []
    ever_forward = False
    previous_forward = 0
    is_fault = bool(item["is_fault"])
    has_behavior = simulator_scenario != "normal"
    behavior_end_s = behavior_start_s + behavior_duration_s
    detection_eligible_value = int(detection_eligible)
    previous_plant_mode: str | None = None
    for source in rows:
        forward = int(source["fdv_actual_forward"])
        plant_mode = str(source["plant_mode"])
        is_cip = plant_mode.startswith("CIP_")
        diversion_event = int(ever_forward and previous_forward and not forward and not is_cip)
        ever_forward = ever_forward or bool(forward)
        previous_forward = forward
        routed_volume = float(source["routed_volume_l"])
        safety_event = int(routed_volume > 1e-12 and not int(source["actual_safe"]))
        simulator_active = int(source["fault_active"])
        behavior_active = (
            0
            if not has_behavior
            else 1
            if simulator_scenario == "cip_cycle"
            else simulator_active
        )
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "time_start_s": source["time_start_s"],
            "time_s": source["time_s"],
            "canonical_code": item["code"],
            "canonical_scenario": item["canonical_scenario"],
            "simulator_scenario": simulator_scenario,
            "fault_active": simulator_active if is_fault else 0,
            "fault_injection_time_s": behavior_start_s if is_fault else "",
            "fault_end_time_s": behavior_end_s if is_fault else "",
            "fault_effect_time_s": (
                effect_time_s
                if is_fault and detection_eligible and effect_time_s is not None
                else ""
            ),
            "behavior_active": behavior_active,
            "behavior_injection_time_s": behavior_start_s if has_behavior else "",
            "behavior_end_time_s": behavior_end_s if has_behavior else "",
            "behavior_effect_time_s": effect_time_s if has_behavior else "",
            "physical_effect_time_s": (
                physical_effect_time_s if has_behavior else ""
            ),
            "detection_eligible": detection_eligible_value,
            "safety_event": safety_event,
            "unsafe_forward_event": int(float(source["unsafe_forward_l"]) > 1e-12),
            "diversion_active": int(not forward and not is_cip),
            "diversion_event": diversion_event,
            "cooling_excursion": int(
                float(source["product_boundary_l"]) > 1e-12
                and float(source["product_temp_c"]) > maximum_product_temp_c
            ),
            "mode_transition_event": int(
                item["kind"] == "mode"
                and previous_plant_mode is not None
                and plant_mode != previous_plant_mode
            ),
        }
        row.update({field: source[field] for field in ORACLE_SOURCE_FIELDS})
        built.append(row)
        previous_plant_mode = plant_mode

    event_fields = [
        "safety_event",
        "unsafe_forward_event",
        "diversion_event",
        "cooling_excursion",
    ]
    onset_by_field = {field: _event_onsets(built, field) for field in event_fields}
    episode_end_s = float(rows[-1]["time_s"]) if rows else 0.0
    for row in built:
        current_time = float(row["time_s"])
        for event_field, onsets in onset_by_field.items():
            for horizon in horizons:
                suffix = f"{horizon:g}s"
                row[f"future_{event_field}_{suffix}"] = (
                    ""
                    if current_time + horizon > episode_end_s + 1e-12
                    else _future_label(onsets, current_time, horizon)
                )
    return built


def _resolved_schema_type(observed: set[str], field: str) -> str:
    if not observed:
        if field.startswith("future_"):
            return "integer"
        if field.endswith("_time_s") or field in {
            "fault_duration_s",
            "fault_severity",
        }:
            return "number"
        return "string"
    if observed <= {"integer", "number"}:
        return "number" if "number" in observed else "integer"
    if len(observed) == 1:
        return next(iter(observed))
    raise ValueError(f"Schema field {field!r} has incompatible types: {sorted(observed)}")


def _schema(
    table_fields: Mapping[str, Sequence[str]],
    observed_types: Mapping[str, Mapping[str, set[str]]],
    nullable: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    context = {
        "episode_id",
        "plant_profile_id",
        "counterfactual_group_id",
        "time_start_s",
        "time_s",
        "step_dt_s",
        "plant_mode",
    }
    def fields_for(table: str, role_for: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": field,
                "type": _resolved_schema_type(
                    set(observed_types[table][field]), field
                ),
                "nullable": bool(nullable[table][field]),
                "role": role_for(field),
            }
            for field in table_fields[table]
        ]

    def label_role(field: str) -> str:
        if field in {"episode_id", "time_start_s", "time_s"}:
            return "join_key"
        if field in EXCLUDED_PRECOMPUTED_DIAGNOSTICS:
            return "excluded_precomputed_diagnostic"
        return "oracle_label"

    def profile_role(field: str) -> str:
        if field == "plant_profile_id":
            return "identity"
        if field == "domain":
            return "domain"
        if field in {"profile_index", "profile_seed", "profile_config_hash"}:
            return "metadata"
        return "parameter"

    return {
        "schema_version": "2.2.0",
        "primary_key": ["episode_id", "time_s"],
        "tables": {
            "signals.csv": {
                "primary_key": ["episode_id", "time_s"],
                "fields": fields_for(
                    "signals.csv",
                    lambda field: "context" if field in context else "observable_feature",
                ),
            },
            "oracle_labels.csv": {
                "primary_key": ["episode_id", "time_s"],
                "fields": fields_for("oracle_labels.csv", label_role),
            },
            "episodes.csv": {
                "primary_key": ["episode_id"],
                "fields": fields_for(
                    "episodes.csv",
                    lambda field: "identity" if field == "episode_id" else "metadata",
                ),
            },
            "profiles.csv": {
                "primary_key": ["plant_profile_id"],
                "fields": fields_for("profiles.csv", profile_role),
            },
        },
    }


def generate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    contract = load_contract()
    ood_contract_path = Path(args.ood_contract)
    ood_contract, domain_ranges = _load_ood_profile_contract(
        ood_contract_path, contract["profile_parameter_ranges"]
    )
    configured_ood_domains = [str(name) for name in ood_contract["ood_domains"]]
    if contract["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"Contract expects model {contract['model_version']}, found {MODEL_VERSION}"
        )
    selected = select_taxonomy(contract, args.scenarios)
    if not selected:
        raise ValueError("At least one implemented scenario is required")
    if any(item["code"] == "F12" for item in selected) and (
        args.duration_s - 60.0 < args.dt_s - 1e-12
    ):
        raise ValueError(
            "F12 incomplete_cleaning requires duration-s to extend beyond "
            "the 60 s CIP pre-rinse"
        )
    implemented = select_taxonomy(contract, None)
    legacy_aliases = {
        alias
        for item in implemented
        for alias in item.get("aliases", [])
        if alias in SCENARIO_NAMES
    }
    contracted_behaviors = {_simulator_name(item) for item in implemented}
    model_behaviors = set(SCENARIO_NAMES) - legacy_aliases
    if contracted_behaviors != model_behaviors:
        raise ValueError(
            "The default ML taxonomy must cover each unique simulator behavior exactly once; "
            f"missing={sorted(model_behaviors - contracted_behaviors)}, "
            f"unknown={sorted(contracted_behaviors - model_behaviors)}"
        )
    _prepare_output(args.output)

    features = feature_fields(contract, "S3-context")
    context_fields = list(contract["signal_context_fields"])
    signal_fields = context_fields + features
    explicit_forbidden = set(contract["always_forbidden_features"]) - set(context_fields)
    accidental = explicit_forbidden & set(signal_fields)
    if accidental:
        raise AssertionError(f"Privileged inputs entered the signal schema: {sorted(accidental)}")
    horizons = [float(value) for value in contract["future_label_horizons_s"]]
    future_fields = [
        f"future_{event}_{h:g}s"
        for event in (
            "safety_event",
            "unsafe_forward_event",
            "diversion_event",
            "cooling_excursion",
        )
        for h in horizons
    ]
    label_fields = LABEL_ID_FIELDS + ORACLE_SOURCE_FIELDS + DERIVED_LABEL_FIELDS + future_fields

    profile_parameter_names = sorted(contract["profile_parameter_ranges"])
    profile_fields = [
        "plant_profile_id",
        "profile_index",
        "profile_seed",
        "profile_config_hash",
        "domain",
        *profile_parameter_names,
    ]
    episode_fields = [
        "episode_id",
        "plant_profile_id",
        "counterfactual_group_id",
        "dataset_version",
        "domain",
        "canonical_code",
        "canonical_scenario",
        "simulator_scenario",
        "implemented",
        "is_fault",
        "replicate",
        "noise_seed",
        "sampling_seed",
        "profile_config_hash",
        "simulator_config_hash",
        "duration_s",
        "dt_s",
        "fault_injection_time_s",
        "fault_end_time_s",
        "fault_duration_s",
        "fault_severity",
        "fault_effect_time_s",
        "behavior_injection_time_s",
        "behavior_end_time_s",
        "behavior_effect_time_s",
        "physical_effect_time_s",
        "detection_eligible",
        "counterfactual_reference",
        "counterfactual_reference_config_hash",
        "row_count",
        "model_version",
        "contract_version",
        "generator_version",
    ]

    profile_path = args.output / "profiles.csv"
    episode_path = args.output / "episodes.csv"
    signal_path = args.output / "signals.csv"
    label_path = args.output / "oracle_labels.csv"
    profile_count = episode_count = signal_count = label_count = 0
    detection_eligible_episode_count = 0
    profile_counts_by_domain: Counter[str] = Counter()
    episode_counts_by_domain: Counter[str] = Counter()
    table_fields = {
        "profiles.csv": profile_fields,
        "episodes.csv": episode_fields,
        "signals.csv": signal_fields,
        "oracle_labels.csv": label_fields,
    }
    observed_types: dict[str, dict[str, set[str]]] = {
        table: {field: set() for field in fields}
        for table, fields in table_fields.items()
    }
    nullable: dict[str, dict[str, bool]] = {
        table: {field: False for field in fields}
        for table, fields in table_fields.items()
    }

    def observe_schema_row(table: str, row: Mapping[str, Any]) -> None:
        for field in table_fields[table]:
            value = row.get(field, "")
            if value is None or value == "":
                nullable[table][field] = True
            else:
                observed_types[table][field].add(field_type(value))

    with (
        profile_path.open("w", newline="", encoding="utf-8") as profile_handle,
        episode_path.open("w", newline="", encoding="utf-8") as episode_handle,
        signal_path.open("w", newline="", encoding="utf-8") as signal_handle,
        label_path.open("w", newline="", encoding="utf-8") as label_handle,
    ):
        profile_writer = write_csv_header(profile_handle, profile_fields)
        episode_writer = write_csv_header(episode_handle, episode_fields)
        signal_writer = write_csv_header(signal_handle, signal_fields)
        label_writer = write_csv_header(label_handle, label_fields)

        profile_specs: list[
            tuple[int, str, str, int, dict[str, float]]
        ] = []
        for profile_index in range(args.profiles):
            profile_id, profile_seed, profile_values = _sample_profile(
                contract, args.seed, profile_index
            )
            profile_specs.append(
                (profile_index, "ID", profile_id, profile_seed, profile_values)
            )
        for ood_index in range(args.ood_profiles):
            domain = configured_ood_domains[ood_index % len(configured_ood_domains)]
            profile_index = args.profiles + ood_index
            profile_id, profile_seed, profile_values = _sample_ood_profile(
                domain_ranges[domain],
                args.seed,
                ood_index,
                profile_index,
                domain,
            )
            profile_specs.append(
                (profile_index, domain, profile_id, profile_seed, profile_values)
            )

        for (
            profile_index,
            domain,
            profile_id,
            profile_seed,
            profile_values,
        ) in profile_specs:
            profile_hash = sha256_json(profile_values)
            profile_row = {
                "plant_profile_id": profile_id,
                "profile_index": profile_index,
                "profile_seed": profile_seed,
                "profile_config_hash": profile_hash,
                "domain": domain,
                **profile_values,
            }
            observe_schema_row("profiles.csv", profile_row)
            profile_writer.writerow(profile_row)
            profile_count += 1
            profile_counts_by_domain[domain] += 1

            for replicate in range(args.replicates):
                sampling_seed, noise_seed, onset_s, fault_duration_s, severity = (
                    _sample_fault_schedule(
                        contract,
                        args.seed,
                        profile_id,
                        replicate,
                        args.duration_s,
                        args.dt_s,
                    )
                )
                common_config = HTSTConfig(
                    **profile_values,
                    duration_s=args.duration_s,
                    dt_s=args.dt_s,
                    random_seed=noise_seed,
                    fault_start_s=onset_s,
                    fault_duration_s=fault_duration_s,
                    fault_severity=severity,
                )
                config_hash = sha256_json(asdict(common_config))
                group_id = f"CF-{profile_id}-R{replicate:03d}-{config_hash[:10]}"
                normal_rows = HTSTSimulator(common_config).run("normal")

                for item in selected:
                    simulator_scenario = _simulator_name(item)
                    scenario_config = common_config
                    scenario_onset_s = onset_s
                    scenario_fault_duration_s = fault_duration_s
                    if simulator_scenario == "incomplete_cleaning":
                        scenario_onset_s, scenario_fault_duration_s = (
                            _sample_cip_fault_schedule(
                                args.seed,
                                profile_id,
                                replicate,
                                args.duration_s,
                                args.dt_s,
                            )
                        )
                        scenario_config = replace(
                            common_config,
                            fault_start_s=scenario_onset_s,
                            fault_duration_s=scenario_fault_duration_s,
                        )
                    scenario_config_hash = sha256_json(asdict(scenario_config))
                    rows = (
                        normal_rows
                        if simulator_scenario == "normal"
                        else HTSTSimulator(scenario_config).run(simulator_scenario)
                    )
                    is_fault = bool(item["is_fault"])
                    has_behavior = simulator_scenario != "normal"
                    behavior_start_s = (
                        0.0
                        if simulator_scenario == "cip_cycle"
                        else scenario_onset_s
                    )
                    behavior_duration_s = (
                        args.duration_s
                        if simulator_scenario == "cip_cycle"
                        else scenario_fault_duration_s
                    )
                    reference_scenario = str(
                        item.get("counterfactual_reference", "normal")
                    )
                    if reference_scenario == "cip_cycle":
                        reference_rows = HTSTSimulator(scenario_config).run(
                            "cip_cycle"
                        )
                    elif reference_scenario == "normal":
                        reference_rows = normal_rows
                    else:
                        raise ValueError(
                            f"Unsupported counterfactual reference {reference_scenario!r} "
                            f"for {item['code']}"
                        )
                    effect_time = (
                        _first_observable_effect(
                            reference_rows,
                            rows,
                            features,
                            behavior_start_s,
                        )
                        if has_behavior
                        else None
                    )
                    observable_effect_required = bool(
                        item.get("observable_effect_required", False)
                    )
                    if has_behavior and effect_time is None and observable_effect_required:
                        raise ValueError(
                            f"{item['code']} ({simulator_scenario}) has no observable v2 effect"
                        )
                    if (
                        effect_time is not None
                        and effect_time + 1e-12 < behavior_start_s
                    ):
                        raise AssertionError(
                            f"{item['code']} effect precedes its scheduled intervention"
                        )
                    physical_effect_time = (
                        _first_observable_effect(
                            reference_rows,
                            rows,
                            ORACLE_SOURCE_FIELDS,
                            behavior_start_s,
                        )
                        if has_behavior
                        else None
                    )
                    if has_behavior and physical_effect_time is None:
                        raise ValueError(
                            f"{item['code']} ({simulator_scenario}) has no physical v2 effect"
                        )
                    detection_eligible = bool(
                        has_behavior
                        and item.get("detection_eligible", True)
                        and effect_time is not None
                    )
                    episode_id = f"{group_id}-{item['code']}"
                    oracle_rows = _build_oracle_rows(
                        rows,
                        item,
                        episode_id,
                        simulator_scenario,
                        behavior_start_s,
                        behavior_duration_s,
                        effect_time,
                        physical_effect_time,
                        detection_eligible,
                        horizons,
                        common_config.maximum_product_temp_c,
                    )
                    for source, oracle in zip(rows, oracle_rows, strict=True):
                        signal = {
                            "episode_id": episode_id,
                            "plant_profile_id": profile_id,
                            "counterfactual_group_id": group_id,
                            **{field: source[field] for field in context_fields[3:]},
                            **{field: source[field] for field in features},
                        }
                        observe_schema_row("signals.csv", signal)
                        observe_schema_row("oracle_labels.csv", oracle)
                        signal_writer.writerow(signal)
                        label_writer.writerow(oracle)
                        signal_count += 1
                        label_count += 1
                    episode_row = {
                            "episode_id": episode_id,
                            "plant_profile_id": profile_id,
                            "counterfactual_group_id": group_id,
                            "dataset_version": args.dataset_version,
                            "domain": domain,
                            "canonical_code": item["code"],
                            "canonical_scenario": item["canonical_scenario"],
                            "simulator_scenario": simulator_scenario,
                            "implemented": int(item["implemented"]),
                            "is_fault": int(is_fault),
                            "replicate": replicate,
                            "noise_seed": noise_seed,
                            "sampling_seed": sampling_seed,
                            "profile_config_hash": profile_hash,
                            "simulator_config_hash": scenario_config_hash,
                            "duration_s": args.duration_s,
                            "dt_s": args.dt_s,
                            "fault_injection_time_s": (
                                scenario_onset_s if is_fault else ""
                            ),
                            "fault_end_time_s": (
                                scenario_onset_s + scenario_fault_duration_s
                                if is_fault
                                else ""
                            ),
                            "fault_duration_s": (
                                scenario_fault_duration_s if is_fault else ""
                            ),
                            "fault_severity": severity if is_fault else "",
                            "fault_effect_time_s": (
                                effect_time
                                if is_fault
                                and detection_eligible
                                and effect_time is not None
                                else ""
                            ),
                            "behavior_injection_time_s": (
                                behavior_start_s if has_behavior else ""
                            ),
                            "behavior_end_time_s": (
                                behavior_start_s + behavior_duration_s
                                if has_behavior
                                else ""
                            ),
                            "behavior_effect_time_s": (
                                effect_time if effect_time is not None else ""
                            ),
                            "physical_effect_time_s": (
                                physical_effect_time
                                if physical_effect_time is not None
                                else ""
                            ),
                            "detection_eligible": int(detection_eligible),
                            "counterfactual_reference": reference_scenario,
                            "counterfactual_reference_config_hash": (
                                scenario_config_hash
                            ),
                            "row_count": len(rows),
                            "model_version": MODEL_VERSION,
                            "contract_version": contract["contract_version"],
                            "generator_version": GENERATOR_VERSION,
                    }
                    observe_schema_row("episodes.csv", episode_row)
                    episode_writer.writerow(episode_row)
                    episode_count += 1
                    episode_counts_by_domain[domain] += 1
                    detection_eligible_episode_count += int(detection_eligible)

    if signal_count != label_count or signal_count == 0:
        raise AssertionError("Signal/label generation produced inconsistent or empty data")

    schema_path = args.output / "dataset_schema.json"
    manifest_path = args.output / "dataset_manifest.json"
    write_json(schema_path, _schema(table_fields, observed_types, nullable))
    scenario_manifest = [
        {
            "code": item["code"],
            "canonical_scenario": item["canonical_scenario"],
            "simulator_scenario": _simulator_name(item),
            "counterfactual_reference": item.get(
                "counterfactual_reference", "normal"
            ),
        }
        for item in selected
    ]
    observed_ood_domains = [
        domain
        for domain in configured_ood_domains
        if profile_counts_by_domain[domain] > 0
    ]
    manifest_profile_ranges = {
        domain: {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in sorted(domain_ranges[domain].items())
        }
        for domain in ["ID", *observed_ood_domains]
    }
    domain_contract = {
        "version": str(ood_contract["version"]),
        "id_domain": "ID",
        "ood_domains": observed_ood_domains,
        "generation_policy": {
            **ood_contract["generation_policy"],
            "configured_ood_domains": configured_ood_domains,
            "active_ood_domains": observed_ood_domains,
        },
        "profile_parameter_ranges": manifest_profile_ranges,
        "source_contract": {
            "filename": ood_contract_path.name,
            "sha256": sha256_file(ood_contract_path),
        },
        "synthetic_only": True,
    }
    manifest = {
        "dataset_version": args.dataset_version,
        "generator_version": GENERATOR_VERSION,
        "contract_version": contract["contract_version"],
        "model_version": MODEL_VERSION,
        "master_seed": args.seed,
        "parameters": {
            "profiles": args.profiles,
            "ood_profiles": args.ood_profiles,
            "total_profiles": args.profiles + args.ood_profiles,
            "replicates": args.replicates,
            "duration_s": args.duration_s,
            "dt_s": args.dt_s,
        },
        "selected_scenarios": scenario_manifest,
        "unimplemented_taxonomy": [
            {
                "code": item["code"],
                "canonical_scenario": item["canonical_scenario"],
                "reason": item.get("reason", "not implemented"),
            }
            for item in contract["taxonomy"]
            if not item["implemented"]
        ],
        "counts": {
            "profiles": profile_count,
            "id_profiles": profile_counts_by_domain["ID"],
            "ood_profiles": sum(
                profile_counts_by_domain[domain]
                for domain in observed_ood_domains
            ),
            "profiles_by_domain": dict(sorted(profile_counts_by_domain.items())),
            "counterfactual_groups": profile_count * args.replicates,
            "episodes": episode_count,
            "episodes_by_domain": dict(sorted(episode_counts_by_domain.items())),
            "detection_eligible_episodes": detection_eligible_episode_count,
            "non_event_or_unobserved_episodes": (
                episode_count - detection_eligible_episode_count
            ),
            "signal_rows": signal_count,
            "label_rows": label_count,
        },
        "signal_context_fields": context_fields,
        "model_feature_fields": features,
        "oracle_label_fields": label_fields,
        "explicitly_forbidden_signal_fields": sorted(explicit_forbidden),
        "domain_contract": domain_contract,
        "provenance": {
            "ml_contract.json": sha256_file(CONTRACT_PATH),
            "ood_profile_contract.json": sha256_file(ood_contract_path),
            "model.py": sha256_file(BASE_DIR / "model.py"),
            "process_components.py": sha256_file(BASE_DIR / "process_components.py"),
            "ml_pipeline_common.py": sha256_file(BASE_DIR / "ml_pipeline_common.py"),
            "generate_ml_dataset.py": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": [
            "profiles.csv",
            "episodes.csv",
            "signals.csv",
            "oracle_labels.csv",
            "dataset_schema.json",
            "dataset_manifest.json",
            "checksums.sha256",
        ],
    }
    write_json(manifest_path, manifest)
    write_checksums(
        args.output,
        [
            "profiles.csv",
            "episodes.csv",
            "signals.csv",
            "oracle_labels.csv",
            "dataset_schema.json",
            "dataset_manifest.json",
        ],
    )
    return args.output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = generate(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    print(
        f"Generated {counts['episodes']} episodes / {counts['signal_rows']} rows "
        f"in {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
