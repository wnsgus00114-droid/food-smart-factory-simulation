#!/usr/bin/env python3
"""Run reproducible HTST scenarios and write auditable research artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from model import (
    ALARM_COLUMNS,
    DIAGNOSTIC_ALARM_COLUMNS,
    MODEL_VERSION,
    HTSTConfig,
    HTSTSimulator,
    SCENARIO_NAMES,
    extract_events,
    summarize,
)
from provenance import source_provenance, staged_output_directory, write_checksums


HERE = Path(__file__).resolve().parent


CONTEXT_FIELDS = {
    "model_version",
    "run_id",
    "config_hash",
    "random_seed",
    "scenario",
    "time_start_s",
    "time_s",
    "step_dt_s",
    "plant_mode",
}
CONTROL_FIELDS = {
    "steam_valve",
    "fdv_command_forward",
    "fdv_command_position",
    "cip_chemical_concentration_pct",
    "cip_cycle_active",
}
OBSERVABLE_SIGNAL_FIELDS = {
    "measured_flow_l_h",
    "maximum_safe_flow_l_h",
    "preheat_temp_sensor_c",
    "control_temp_sensor_c",
    "safety_temp_sensor_c",
    "sensor_disagreement_c",
    "estimated_residence_time_s",
    "estimated_fastest_residence_time_s",
    "raw_pressure_sensor_bar",
    "pasteurized_pressure_sensor_bar",
    "measured_differential_pressure_bar",
    "booster_pump_speed_fraction",
    "leak_detector_signal_fraction",
    "product_temp_sensor_c",
    "fdv_position_feedback",
    "fdv_position_error",
    "fdv_mismatch_time_s",
    "cip_conductivity_proxy_ms_cm",
    "cip_ph_proxy",
    "cip_release_permissive",
    "post_cip_conductivity_proxy_ms_cm",
    "post_cip_ph_proxy",
    "restart_product_interface_signal_fraction",
    "power_good_signal",
    "temperature_sensor_quality_ok",
}
ORACLE_FIELDS = {
    "fault_active",
    "flow_l_h",
    "inlet_temp_c",
    "preheat_temp_c",
    "heater_out_temp_c",
    "holding_out_temp_c",
    "actual_residence_time_s",
    "fastest_residence_time_s",
    "mean_transport_residence_time_s",
    "fastest_flow_efficiency",
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
    "raw_pressure_bar",
    "pasteurized_pressure_bar",
    "differential_pressure_bar",
    "booster_pump_factor",
    "leak_fraction",
    "contamination_risk",
    "fouling_index",
    "surface_protein_soil_g",
    "surface_mineral_soil_g",
    "surface_total_soil_g",
    "surface_hygiene_risk_fraction",
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
    "cip_effective_chemical_concentration_pct",
    "cip_protein_soil_g",
    "cip_mineral_soil_g",
    "cip_total_soil_g",
    "cip_removed_protein_g",
    "cip_removed_mineral_g",
    "cip_residual_chemical_fraction",
    "cip_cleaning_complete",
    "actual_safe",
    "product_temp_c",
    "potential_product_temp_c",
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
    "fdv_actual_forward",
    "fdv_position",
    "fdv_forward_fraction",
    "product_downstream_fault_fraction",
    "product_regenerator_factor",
    "product_cooler_factor",
    "sensor_available",
    "power_available",
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
    "shadow_cooler_product_out_c",
    "shadow_cooler_utility_out_c",
    "shadow_cooler_wall_temp_c",
    "shadow_cooler_effective_ua_kw_k",
    "shadow_cooler_energy_balance_error_kj",
    "alarm_unsafe_forward",
    "alarm_high_fouling",
    "diagnostic_alarm_count",
    "oracle_alarm_count",
    "bounded_exponential_count",
    "bounded_exponential_diagnostic",
    "diagnostic_count",
    "total_alarm_count",
    "control_sensor_bias_c",
    "safety_sensor_bias_c",
    "flow_meter_bias_fraction",
    "heater_capacity_factor",
    "regenerator_factor",
    "cooler_factor",
    "valve_travel_time_factor",
    "valve_leakage_fraction",
    "cleaning_effectiveness_factor",
    "mass_balance_error_l",
    "post_fdv_balance_error_l",
    "balance_tank_volume_error_l",
    "balance_tank_temperature_moment_error_l_c",
    "balance_tank_pass_moment_error_l",
    "balance_tank_risk_volume_error_l",
    "balance_tank_chemical_volume_error_l",
    "balance_tank_product_volume_error_l",
    "balance_tank_ambient_heat_kj",
}
OUTCOME_FIELDS = {
    "routed_volume_l",
    "forward_l",
    "valve_forward_l",
    "post_fdv_inlet_l",
    "product_boundary_l",
    "safe_forward_l",
    "diverted_l",
    "cip_recirculated_l",
    "fresh_feed_l",
    "return_to_balance_tank_l",
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
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def field_role(name: str) -> str:
    if name in CONTEXT_FIELDS:
        return "context"
    if name in CONTROL_FIELDS:
        return "control"
    if name in ORACLE_FIELDS:
        return "oracle"
    if name in OUTCOME_FIELDS:
        return "outcome"
    if name in DIAGNOSTIC_ALARM_COLUMNS:
        return "oracle"
    if name in ALARM_COLUMNS or name in {"alarm_count", "observable_alarm_count"}:
        return "observable_alarm"
    if name in OBSERVABLE_SIGNAL_FIELDS:
        return "observable_signal"
    raise ValueError(
        f"Unclassified simulator field {name!r}; assign an explicit schema role"
    )


def field_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def schema_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "2.2.0",
        "model_version": MODEL_VERSION,
        "field_count": len(row),
        "roles": {
            "observable_signal": "May be considered as an ML feature if available in the target plant.",
            "observable_alarm": "PLC-derived alarm; use only in experiments that explicitly allow alarms.",
            "control": "Known command or actuator state.",
            "context": "Episode identity, time, mode or reproducibility metadata.",
            "oracle": "Simulation truth or injected fault information; forbidden as an ML input.",
            "outcome": "Accumulated/routed result; label or evaluation only, not an online input.",
        },
        "fields": [
            {"name": name, "type": field_type(value), "role": field_role(name)}
            for name, value in row.items()
        ],
    }


def validate_rows_against_schema(
    rows: list[dict[str, object]], schema_row: dict[str, object]
) -> None:
    """Reject scenario output whose columns or scalar types drift from the schema."""

    expected_names = list(schema_row)
    expected_types = {
        name: field_type(value) for name, value in schema_row.items()
    }
    for row_index, row in enumerate(rows):
        if list(row) != expected_names:
            raise ValueError(
                f"schema column drift at row {row_index}: "
                f"expected {expected_names}, got {list(row)}"
            )
        for name, value in row.items():
            expected = expected_types[name]
            observed = field_type(value)
            if observed == expected or (expected == "number" and observed == "integer"):
                continue
            raise ValueError(
                f"schema type drift at row {row_index}, field {name!r}: "
                f"expected {expected}, got {observed}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_NAMES),
        choices=SCENARIO_NAMES,
    )
    parser.add_argument("--duration", type=float, default=900.0, help="seconds")
    parser.add_argument("--dt", type=float, default=0.5, help="integration step, seconds")
    parser.add_argument("--seed", type=int, default=HTSTConfig.random_seed)
    parser.add_argument("--fault-start", type=float, default=HTSTConfig.fault_start_s)
    parser.add_argument("--fault-duration", type=float, default=HTSTConfig.fault_duration_s)
    parser.add_argument("--severity", type=float, default=HTSTConfig.fault_severity)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios must not contain duplicates")
    args.output = args.output.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = replace(
            HTSTConfig(),
            duration_s=args.duration,
            dt_s=args.dt,
            random_seed=args.seed,
            fault_start_s=args.fault_start,
            fault_duration_s=args.fault_duration,
            fault_severity=args.severity,
        )
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    fault_scenarios = {
        scenario
        for scenario in args.scenarios
        if scenario not in {"normal", "cip_cycle"}
    }
    if (
        fault_scenarios
        and config.fault_severity > 0.0
        and (
            config.fault_duration_s <= 0.0
            or config.fault_start_s >= config.duration_s
        )
    ):
        names = ", ".join(sorted(fault_scenarios))
        raise SystemExit(
            "configuration error: the fault interval does not overlap the run "
            f"for scenario(s): {names}"
        )

    summaries: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    first_row: dict[str, object] | None = None
    run_ids: dict[str, str] = {}
    source = source_provenance(
        {
            "model.py": HERE / "model.py",
            "process_components.py": HERE / "process_components.py",
            "provenance.py": HERE / "provenance.py",
            "run_scenarios.py": HERE / "run_scenarios.py",
        }
    )

    with staged_output_directory(args.output) as staging:
        generated: list[Path] = []
        for scenario in args.scenarios:
            rows = HTSTSimulator(config).run(scenario)
            if not rows:
                raise ValueError(f"scenario {scenario!r} produced no records")
            first_row = first_row or rows[0]
            validate_rows_against_schema(rows, first_row)
            run_ids[scenario] = str(rows[0]["run_id"])
            csv_path = staging / f"{scenario}.csv"
            write_csv(csv_path, rows)
            generated.append(csv_path)
            summaries.append(summarize(rows, config))
            for event in extract_events(rows):
                all_events.append(
                    {
                        "run_id": rows[0]["run_id"],
                        "scenario": scenario,
                        "role": (
                            "oracle"
                            if event["event_name"] == "alarm_unsafe_forward"
                            else "observable_alarm"
                            if event["event_type"] == "ALARM"
                            else "context"
                        ),
                        **event,
                    }
                )

        if all_events:
            events_path = staging / "event_log.csv"
            write_csv(events_path, all_events)
            generated.append(events_path)

        summary_path = staging / "scenario_summary.json"
        write_json(
            summary_path,
            {
                "model_status": "unvalidated engineering research surrogate",
                "model_version": MODEL_VERSION,
                "scenarios": summaries,
            },
        )
        generated.append(summary_path)

        schema_path = staging / "schema.json"
        write_json(schema_path, schema_from_row(first_row or {}))
        generated.append(schema_path)

        manifest_path = staging / "run_manifest.json"
        write_json(
            manifest_path,
            {
                "model_status": "unvalidated engineering research surrogate",
                "model_version": MODEL_VERSION,
                "python_version": platform.python_version(),
                "config": asdict(config),
                "config_hash": first_row["config_hash"] if first_row else None,
                "run_ids": run_ids,
                "scenarios": list(args.scenarios),
                "source_provenance": source,
                "artifacts": [path.name for path in generated],
                "reproduce": (
                    "python3 run_scenarios.py "
                    f"--duration {args.duration} --dt {args.dt} --seed {args.seed} "
                    f"--fault-start {args.fault_start} --fault-duration {args.fault_duration} "
                    f"--severity {args.severity} --scenarios {' '.join(args.scenarios)}"
                ),
            },
        )
        generated.append(manifest_path)
        write_checksums(staging, generated)

    print(
        "scenario\tdiverted_fraction\tunsafe_forward_l\t"
        "min_forward_temp_c\tmax_product_temp_c"
    )
    for item in summaries:
        print(
            f'{item["scenario"]}\t{item["diverted_fraction"]:.4f}\t'
            f'{item["unsafe_forward_l"]:.3f}\t'
            f'{item["minimum_forward_holding_temp_c"]}\t'
            f'{item["maximum_forward_product_temp_c"]}'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
