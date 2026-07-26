#!/usr/bin/env python3
"""Generate isolated long-horizon scalar-fouling health/RUL trajectories."""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        EPSILON,
        evaluate_eol_trace,
        feature_fields,
        field_declaration,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        rolling_energy_intensity_ratios,
        select_sample_indices,
        sha256_file,
        sha256_json,
        stable_seed,
        write_checksums,
        write_csv_header,
        write_json,
        write_schema,
    )
    from .model import HTSTConfig, HTSTSimulator, MODEL_VERSION
except ImportError:  # pragma: no cover - direct CLI execution
    from d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        EPSILON,
        evaluate_eol_trace,
        feature_fields,
        field_declaration,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        rolling_energy_intensity_ratios,
        select_sample_indices,
        sha256_file,
        sha256_json,
        stable_seed,
        write_checksums,
        write_csv_header,
        write_json,
        write_schema,
    )
    from model import HTSTConfig, HTSTSimulator, MODEL_VERSION


GENERATOR_VERSION = "1.2.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", default="D3-RUL-pilot")
    parser.add_argument("--profiles", type=int, default=80)
    parser.add_argument("--trajectories-per-profile", type=int, default=10)
    parser.add_argument("--ood-profiles", type=int, default=0)
    parser.add_argument("--dt-s", type=float)
    parser.add_argument("--sample-interval-s", type=float)
    parser.add_argument("--maximum-sim-duration-s", type=float)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--accelerations", type=float, nargs="+")
    parser.add_argument(
        "--censor-horizons-equivalent-h", type=float, nargs="+"
    )
    return parser


def _rotated_choice(values: Sequence[float], seed: int, position: int) -> float:
    if not values:
        raise ValueError("choice list cannot be empty")
    offset = random.Random(seed).randrange(len(values))
    return float(values[(offset + position) % len(values)])


def _sample_profile(
    contract: Mapping[str, Any], master_seed: int, index: int, domain: str
) -> tuple[str, int, dict[str, float]]:
    if domain not in {"ID", "OOD"}:
        raise ValueError(f"unsupported D3 profile domain: {domain!r}")
    seed = stable_seed(master_seed, "d3-profile", index)
    rng = random.Random(seed)
    ranges = dict(contract["profile_parameter_ranges"])
    if domain == "OOD":
        ranges.update(contract["ood_profile_parameter_ranges"])
    values = {
        name: rng.uniform(float(bounds[0]), float(bounds[1]))
        for name, bounds in sorted(ranges.items())
    }
    profile_hash = sha256_json(values)
    return f"D3P{index:04d}-{profile_hash[:10]}", seed, values


def _profile_config(profile: Mapping[str, float]) -> dict[str, float]:
    return {
        name: value
        for name, value in profile.items()
        if name != "base_fouling_rate_per_h"
    }


def _schema_tables(
    profile_fields: Sequence[str],
    trajectory_fields: Sequence[str],
    signal_fields: Sequence[str],
    label_fields: Sequence[str],
    evidence_fields: Sequence[str],
    signal_context_fields: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    numeric_profile = set(profile_fields) - {
        "plant_profile_id",
        "profile_config_hash",
        "domain",
    }
    integer_profile = {"profile_index", "profile_seed"}
    trajectory_strings = {
        "trajectory_id",
        "life_family_id",
        "plant_profile_id",
        "profile_config_hash",
        "domain",
        "outcome_type",
        "eol_cause",
        "eol_cause_mask",
        "actual_config_hash",
        "reference_config_hash",
        "model_version",
        "contract_version",
        "generator_version",
    }
    trajectory_integers = {
        "trajectory_index",
        "noise_seed",
        "degradation_seed",
        "censor_seed",
        "event_observed",
        "row_count",
    }
    label_strings = {"trajectory_id", "eol_cause", "eol_cause_mask"}
    label_integers = {
        "event_observed",
        "terminal_event",
        "maintenance_needed",
        "fdv_actual_forward",
        "eol_monitor_armed",
    }
    evidence_strings = {"trajectory_id", "condition"}
    nullable_trajectory = {
        "eol_time_sim_s",
        "eol_time_equivalent_s",
        "eol_cause",
        "eol_cause_mask",
        "censor_time_sim_s",
        "censor_time_equivalent_s",
    }
    nullable_labels = {
        "eol_cause",
        "eol_cause_mask",
        "actual_energy_intensity_ratio",
        "rul_sim_s",
        "rul_equivalent_s",
        "rul_lower_bound_sim_s",
        "rul_lower_bound_equivalent_s",
    }

    def declaration(
        name: str,
        strings: set[str],
        integers: set[str],
        nullable: set[str],
        role: str,
    ) -> dict[str, Any]:
        declared = "string" if name in strings else "integer" if name in integers else "number"
        return field_declaration(name, declared, name in nullable, role)

    signal_context = set(signal_context_fields)
    return {
        "profiles.csv": [
            field_declaration(
                name,
                "integer" if name in integer_profile else "number" if name in numeric_profile else "string",
                False,
                "profile_metadata",
            )
            for name in profile_fields
        ],
        "trajectories.csv": [
            declaration(
                name,
                trajectory_strings,
                trajectory_integers,
                nullable_trajectory,
                "trajectory_metadata",
            )
            for name in trajectory_fields
        ],
        "signals.csv": [
            field_declaration(
                name,
                "string" if name in {"trajectory_id", "plant_profile_id", "life_family_id"} else "number",
                False,
                "context" if name in signal_context else "observable_feature",
            )
            for name in signal_fields
        ],
        "oracle_labels.csv": [
            declaration(
                name,
                label_strings,
                label_integers,
                nullable_labels,
                "oracle_label" if name not in {"trajectory_id", "time_start_sim_s", "time_sim_s"} else "join_key",
            )
            for name in label_fields
        ],
        "eol_evidence.csv": [
            declaration(name, evidence_strings, set(), set(), "oracle_evidence")
            for name in evidence_fields
        ],
    }


def generate(args: argparse.Namespace) -> Path:
    contract = load_d3_contract()
    if not model_version_is_compatible(MODEL_VERSION, contract):
        raise ValueError(
            f"D3 contract does not allow runtime model version {MODEL_VERSION}"
        )
    dt_s = float(args.dt_s or contract["default_dt_s"])
    sample_interval_s = float(
        args.sample_interval_s or contract["default_sample_interval_s"]
    )
    maximum_sim_duration_s = float(
        args.maximum_sim_duration_s or contract["maximum_sim_duration_s"]
    )
    accelerations = list(
        args.accelerations or contract["degradation_acceleration_factors"]
    )
    horizons_h = list(
        args.censor_horizons_equivalent_h
        or contract["administrative_censor_horizons_equivalent_h"]
    )
    if args.profiles <= 0 or args.trajectories_per_profile <= 0:
        raise ValueError("profiles and trajectories-per-profile must be positive")
    if args.ood_profiles < 0 or args.ood_profiles >= args.profiles:
        raise ValueError("ood-profiles must be in [0, profiles)")
    if not 0.0 < dt_s <= 1.0:
        raise ValueError("dt-s must be in (0, 1]")
    if not math.isfinite(sample_interval_s) or sample_interval_s + EPSILON < dt_s:
        raise ValueError("sample-interval-s must be at least dt-s")
    if not math.isfinite(maximum_sim_duration_s) or maximum_sim_duration_s <= 0.0:
        raise ValueError("maximum-sim-duration-s must be positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in accelerations + horizons_h):
        raise ValueError("accelerations and censor horizons must be finite and positive")
    prepare_empty_output(args.output)

    profile_parameter_names = sorted(contract["profile_parameter_ranges"])
    profile_fields = [
        "plant_profile_id",
        "profile_index",
        "profile_seed",
        "profile_config_hash",
        "domain",
        *profile_parameter_names,
    ]
    trajectory_fields = [
        "trajectory_id",
        "life_family_id",
        "trajectory_index",
        "plant_profile_id",
        "profile_config_hash",
        "domain",
        "noise_seed",
        "degradation_seed",
        "censor_seed",
        "degradation_acceleration_factor",
        "base_fouling_rate_per_h",
        "accelerated_fouling_rate_per_h",
        "requested_censor_horizon_equivalent_h",
        "planned_sim_duration_s",
        "administrative_censor_time_sim_s",
        "administrative_censor_time_equivalent_s",
        "observation_end_sim_s",
        "observation_end_equivalent_s",
        "event_observed",
        "outcome_type",
        "eol_time_sim_s",
        "eol_time_equivalent_s",
        "eol_cause",
        "eol_cause_mask",
        "censor_time_sim_s",
        "censor_time_equivalent_s",
        "row_count",
        "actual_config_hash",
        "reference_config_hash",
        "model_version",
        "contract_version",
        "generator_version",
    ]
    context_fields = list(contract["signal_context_fields"])
    signal_feature_fields = feature_fields(contract)
    signal_fields = context_fields + signal_feature_fields
    label_fields = [
        "trajectory_id",
        "time_start_sim_s",
        "time_sim_s",
        "fouling_index",
        "health_index",
        "maintenance_needed",
        "fdv_actual_forward",
        "actual_energy_intensity_ratio",
        "eol_monitor_armed",
        "event_observed",
        "terminal_event",
        "eol_cause",
        "eol_cause_mask",
        "time_to_event_or_censor_sim_s",
        "time_to_event_or_censor_equivalent_s",
        "rul_sim_s",
        "rul_equivalent_s",
        "rul_lower_bound_sim_s",
        "rul_lower_bound_equivalent_s",
    ]
    evidence_fields = [
        "trajectory_id",
        "condition",
        "persistence_start_sim_s",
        "trigger_time_sim_s",
        "trigger_time_equivalent_s",
        "trigger_value",
    ]

    paths = {
        "profiles.csv": args.output / "profiles.csv",
        "trajectories.csv": args.output / "trajectories.csv",
        "signals.csv": args.output / "signals.csv",
        "oracle_labels.csv": args.output / "oracle_labels.csv",
        "eol_evidence.csv": args.output / "eol_evidence.csv",
    }
    counts = {
        "profiles": 0,
        "trajectories": 0,
        "signal_rows": 0,
        "label_rows": 0,
        "eol_evidence_rows": 0,
        "events": 0,
        "right_censored": 0,
    }
    with (
        paths["profiles.csv"].open("w", newline="", encoding="utf-8") as profile_handle,
        paths["trajectories.csv"].open("w", newline="", encoding="utf-8") as trajectory_handle,
        paths["signals.csv"].open("w", newline="", encoding="utf-8") as signal_handle,
        paths["oracle_labels.csv"].open("w", newline="", encoding="utf-8") as label_handle,
        paths["eol_evidence.csv"].open("w", newline="", encoding="utf-8") as evidence_handle,
    ):
        profile_writer = write_csv_header(profile_handle, profile_fields)
        trajectory_writer = write_csv_header(trajectory_handle, trajectory_fields)
        signal_writer = write_csv_header(signal_handle, signal_fields)
        label_writer = write_csv_header(label_handle, label_fields)
        evidence_writer = write_csv_header(evidence_handle, evidence_fields)

        for profile_index in range(args.profiles):
            domain = "OOD" if profile_index >= args.profiles - args.ood_profiles else "ID"
            profile_id, profile_seed, profile = _sample_profile(
                contract, args.seed, profile_index, domain
            )
            profile_hash = sha256_json(profile)
            profile_writer.writerow(
                {
                    "plant_profile_id": profile_id,
                    "profile_index": profile_index,
                    "profile_seed": profile_seed,
                    "profile_config_hash": profile_hash,
                    "domain": domain,
                    **profile,
                }
            )
            counts["profiles"] += 1

            for trajectory_index in range(args.trajectories_per_profile):
                life_family_id = f"LF-{profile_id}-T{trajectory_index:03d}"
                degradation_seed = stable_seed(
                    args.seed, "d3-degradation", profile_id, trajectory_index
                )
                censor_seed = stable_seed(
                    args.seed, "d3-censor", profile_id, trajectory_index
                )
                noise_seed = stable_seed(
                    args.seed, "d3-noise", profile_id, trajectory_index
                )
                acceleration = _rotated_choice(
                    accelerations, degradation_seed, trajectory_index
                )
                requested_horizon_h = _rotated_choice(
                    horizons_h, censor_seed, trajectory_index
                )
                raw_duration_s = requested_horizon_h * 3600.0 / acceleration
                duration_s = min(
                    maximum_sim_duration_s,
                    max(dt_s, math.ceil(raw_duration_s / dt_s - EPSILON) * dt_s),
                )
                base_rate = float(profile["base_fouling_rate_per_h"])
                accelerated_rate = base_rate * acceleration
                common = {
                    **_profile_config(profile),
                    "duration_s": duration_s,
                    "dt_s": dt_s,
                    "random_seed": noise_seed,
                    "maximum_steps": int(math.ceil(duration_s / dt_s)) + 10,
                }
                actual_config = HTSTConfig(
                    **common,
                    fouling_rate_per_h=accelerated_rate,
                )
                reference_config = HTSTConfig(
                    **{
                        **common,
                        "initial_fouling_index": 0.0,
                    },
                    fouling_rate_per_h=0.0,
                )
                trajectory_id = (
                    f"D3T-{profile_id}-{trajectory_index:03d}-"
                    f"{sha256_json(asdict(actual_config))[:10]}"
                )
                actual_rows = HTSTSimulator(actual_config).run("normal")
                reference_rows = HTSTSimulator(reference_config).run("normal")
                ratios = rolling_energy_intensity_ratios(
                    actual_rows,
                    reference_rows,
                    float(contract["eol"]["energy_window_sim_s"]),
                )
                # EOL is deliberately evaluated on the exact rows exported to
                # signals/oracle_labels.  The exported intervals are rebuilt as
                # a contiguous partition of [0, observation_end], so an auditor
                # needs neither the hidden model-rate trace nor generator state
                # to reproduce arming, persistence, and the first trigger.
                candidate_indices = select_sample_indices(
                    actual_rows,
                    float(actual_rows[-1]["time_s"]),
                    sample_interval_s,
                )
                eol_rows: list[dict[str, Any]] = []
                previous_sample_time_s = 0.0
                for index in candidate_indices:
                    row = actual_rows[index]
                    time_s = float(row["time_s"])
                    step_dt_sim_s = time_s - previous_sample_time_s
                    eol_rows.append(
                        {
                            "time_start_sim_s": previous_sample_time_s,
                            "time_sim_s": time_s,
                            "step_dt_sim_s": step_dt_sim_s,
                            "fouling_index": row["fouling_index"],
                            "steam_valve": row["steam_valve"],
                            "fdv_actual_forward": row["fdv_actual_forward"],
                            "actual_energy_intensity_ratio": ratios[index],
                        }
                    )
                    previous_sample_time_s = time_s
                eol_trace, evidence, outcome = evaluate_eol_trace(
                    eol_rows, acceleration, contract["eol"]
                )
                event_observed = outcome is not None
                observation_end_s = (
                    float(outcome["eol_time_sim_s"])
                    if outcome is not None
                    else float(actual_rows[-1]["time_s"])
                )
                observation_end_equivalent_s = observation_end_s * acceleration
                terminal_position = next(
                    position
                    for position, item in enumerate(eol_rows)
                    if math.isclose(
                        float(item["time_sim_s"]),
                        observation_end_s,
                        abs_tol=EPSILON,
                    )
                )
                sample_indices = candidate_indices[: terminal_position + 1]
                emitted_eol_rows = eol_rows[: terminal_position + 1]
                emitted_eol_trace = eol_trace[: terminal_position + 1]
                event_cause = str(outcome["eol_cause"]) if outcome else ""
                event_cause_mask = str(outcome["eol_cause_mask"]) if outcome else ""
                trajectory_writer.writerow(
                    {
                        "trajectory_id": trajectory_id,
                        "life_family_id": life_family_id,
                        "trajectory_index": trajectory_index,
                        "plant_profile_id": profile_id,
                        "profile_config_hash": profile_hash,
                        "domain": domain,
                        "noise_seed": noise_seed,
                        "degradation_seed": degradation_seed,
                        "censor_seed": censor_seed,
                        "degradation_acceleration_factor": acceleration,
                        "base_fouling_rate_per_h": base_rate,
                        "accelerated_fouling_rate_per_h": accelerated_rate,
                        "requested_censor_horizon_equivalent_h": requested_horizon_h,
                        "planned_sim_duration_s": duration_s,
                        "administrative_censor_time_sim_s": duration_s,
                        "administrative_censor_time_equivalent_s": duration_s * acceleration,
                        "observation_end_sim_s": observation_end_s,
                        "observation_end_equivalent_s": observation_end_equivalent_s,
                        "event_observed": int(event_observed),
                        "outcome_type": "EVENT" if event_observed else "RIGHT_CENSORED",
                        "eol_time_sim_s": observation_end_s if event_observed else "",
                        "eol_time_equivalent_s": observation_end_equivalent_s if event_observed else "",
                        "eol_cause": event_cause,
                        "eol_cause_mask": event_cause_mask,
                        "censor_time_sim_s": "" if event_observed else observation_end_s,
                        "censor_time_equivalent_s": "" if event_observed else observation_end_equivalent_s,
                        "row_count": len(sample_indices),
                        "actual_config_hash": sha256_json(asdict(actual_config)),
                        "reference_config_hash": sha256_json(asdict(reference_config)),
                        "model_version": MODEL_VERSION,
                        "contract_version": contract["contract_version"],
                        "generator_version": GENERATOR_VERSION,
                    }
                )
                counts["trajectories"] += 1
                counts["events" if event_observed else "right_censored"] += 1

                utility_noise = float(contract["utility_sensor_relative_noise_std"])
                for sample_position, index in enumerate(sample_indices):
                    row = actual_rows[index]
                    eol_row = emitted_eol_rows[sample_position]
                    time_s = float(row["time_s"])
                    time_start_s = float(eol_row["time_start_sim_s"])
                    step_dt = float(eol_row["step_dt_sim_s"])
                    measurement_rng = random.Random(
                        stable_seed(noise_seed, "d3-utility", f"{time_s:.9f}")
                    )
                    heater_sensor = max(
                        0.0,
                        float(row["heat_kw"])
                        * (1.0 + measurement_rng.gauss(0.0, utility_noise)),
                    )
                    pump_sensor = max(
                        0.0,
                        float(row["pump_power_kw"])
                        * (1.0 + measurement_rng.gauss(0.0, utility_noise)),
                    )
                    signal_values = {
                        "control_temp_sensor_c": row["control_temp_sensor_c"],
                        "safety_temp_sensor_c": row["safety_temp_sensor_c"],
                        "preheat_temp_sensor_c": row["preheat_temp_sensor_c"],
                        "product_temp_sensor_c": row["product_temp_sensor_c"],
                        "measured_flow_l_h": row["measured_flow_l_h"],
                        "steam_valve": row["steam_valve"],
                        "raw_pressure_sensor_bar": row["raw_pressure_sensor_bar"],
                        "pasteurized_pressure_sensor_bar": row["pasteurized_pressure_sensor_bar"],
                        "measured_differential_pressure_bar": row["measured_differential_pressure_bar"],
                        "booster_pump_speed_fraction": row["booster_pump_speed_fraction"],
                        "fdv_command_position": row["fdv_command_position"],
                        "fdv_position_feedback": row["fdv_position_feedback"],
                        "heater_power_sensor_kw": heater_sensor,
                        "pump_power_sensor_kw": pump_sensor,
                        "power_good_signal": row["power_good_signal"],
                        "temperature_sensor_quality_ok": row["temperature_sensor_quality_ok"],
                        "operating_time_meter_h": time_s / 3600.0,
                    }
                    signal_writer.writerow(
                        {
                            "trajectory_id": trajectory_id,
                            "plant_profile_id": profile_id,
                            "life_family_id": life_family_id,
                            "time_start_sim_s": time_start_s,
                            "time_sim_s": time_s,
                            "step_dt_sim_s": step_dt,
                            **signal_values,
                        }
                    )
                    remaining_sim_s = max(0.0, observation_end_s - time_s)
                    remaining_equivalent_s = remaining_sim_s * acceleration
                    terminal = math.isclose(time_s, observation_end_s, abs_tol=EPSILON)
                    ratio = ratios[index]
                    label_writer.writerow(
                        {
                            "trajectory_id": trajectory_id,
                            "time_start_sim_s": time_start_s,
                            "time_sim_s": time_s,
                            "fouling_index": row["fouling_index"],
                            "health_index": max(0.0, 1.0 - float(row["fouling_index"])),
                            "maintenance_needed": int(
                                float(row["fouling_index"])
                                >= float(contract["maintenance_fouling_index"])
                            ),
                            "fdv_actual_forward": row["fdv_actual_forward"],
                            "actual_energy_intensity_ratio": "" if ratio is None else ratio,
                            "eol_monitor_armed": emitted_eol_trace[sample_position][
                                "eol_monitor_armed"
                            ],
                            "event_observed": int(event_observed),
                            "terminal_event": int(event_observed and terminal),
                            "eol_cause": event_cause,
                            "eol_cause_mask": event_cause_mask,
                            "time_to_event_or_censor_sim_s": remaining_sim_s,
                            "time_to_event_or_censor_equivalent_s": remaining_equivalent_s,
                            "rul_sim_s": remaining_sim_s if event_observed else "",
                            "rul_equivalent_s": remaining_equivalent_s if event_observed else "",
                            "rul_lower_bound_sim_s": "" if event_observed else remaining_sim_s,
                            "rul_lower_bound_equivalent_s": "" if event_observed else remaining_equivalent_s,
                        }
                    )
                    counts["signal_rows"] += 1
                    counts["label_rows"] += 1

                for condition in contract["eol"]["cause_priority"]:
                    item = evidence.get(condition)
                    if item is None or float(item["trigger_time_sim_s"]) > observation_end_s + EPSILON:
                        continue
                    evidence_writer.writerow(
                        {
                            "trajectory_id": trajectory_id,
                            **item,
                        }
                    )
                    counts["eol_evidence_rows"] += 1

    schema_tables = _schema_tables(
        profile_fields,
        trajectory_fields,
        signal_fields,
        label_fields,
        evidence_fields,
        contract["signal_context_fields"],
    )
    write_schema(
        args.output / "dataset_schema.json",
        contract["contract_version"],
        schema_tables,
    )
    manifest = {
        "dataset_family": "D3-RUL",
        "dataset_version": args.dataset_version,
        "contract_version": contract["contract_version"],
        "generator_version": GENERATOR_VERSION,
        "model_version": MODEL_VERSION,
        "model_compatibility_policy": {
            "allowed_major_versions": contract["compatible_model_major_versions"],
            "runtime_version_recorded_exactly": True,
        },
        "seed": args.seed,
        "parameters": {
            "profiles": args.profiles,
            "trajectories_per_profile": args.trajectories_per_profile,
            "ood_profiles": args.ood_profiles,
            "dt_s": dt_s,
            "sample_interval_s": sample_interval_s,
            "eol_evaluation_resolution_sim_s": sample_interval_s,
            "maximum_sim_duration_s": maximum_sim_duration_s,
            "accelerations": accelerations,
            "censor_horizons_equivalent_h": horizons_h,
            "ood_definition": contract["ood_definition"],
            "ood_override_fields": sorted(
                contract["ood_profile_parameter_ranges"]
            ),
        },
        "feature_set": "H1-age",
        "model_feature_fields": signal_feature_fields,
        "right_censoring_policy": "Administrative horizon is sampled from a dedicated censor seed before simulation; censored RUL is null and only a lower bound is emitted.",
        "counts": counts,
        "provenance": {
            "d3_rul_contract.json": sha256_file(CONTRACT_PATH),
            "d3_rul_common.py": sha256_file(BASE_DIR / "d3_rul_common.py"),
            "generate_d3_rul_dataset.py": sha256_file(Path(__file__).resolve()),
            "model.py": sha256_file(BASE_DIR / "model.py"),
        },
        "artifacts": [
            *paths,
            "dataset_schema.json",
            "dataset_manifest.json",
            "checksums.sha256",
        ],
    }
    write_json(args.output / "dataset_manifest.json", manifest)
    write_checksums(
        args.output,
        [
            *paths,
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
    manifest = (output / "dataset_manifest.json").resolve()
    print(f"Generated D3-RUL dataset: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
