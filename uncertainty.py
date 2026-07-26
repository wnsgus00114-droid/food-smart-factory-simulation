#!/usr/bin/env python3
"""Reproducible Monte Carlo and one-at-a-time analysis for the HTST model.

Only the Python standard library is required.  Monte Carlo samples are paired
across scenarios: a given sample index uses the same process configuration and
model seed for every selected scenario, plus a mode-matched counterfactual
(normal production or normal CIP).  OAT sensitivity uses common random numbers
so that a parameter effect is not confounded with sensor-noise draws.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

from model import (
    ALARM_COLUMNS,
    MODEL_VERSION,
    HTSTConfig,
    HTSTSimulator,
    SCENARIO_NAMES,
    summarize,
)
from provenance import source_provenance, staged_output_directory, write_checksums


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "uncertainty_results"
DEFAULT_SCENARIOS = (
    "steam_loss",
    "flow_surge",
    "dual_sensor_common_bias",
    "booster_pump_failure",
    "progressive_fouling",
    "valve_stuck_forward_steam_loss",
    "cooling_utility_loss",
    "power_failure",
    "start_stop",
    "incomplete_cleaning",
    "sensor_drift",
    "sensor_dropout",
    "slow_valve",
    "valve_leakage",
)

# Scalars retained from ``summarize`` for run-level analysis.  Event-window
# fields are added by ``_fault_window_metrics`` below.
SUMMARY_METRICS = (
    "routed_l",
    "forward_l",
    "diverted_l",
    "diverted_fraction",
    "fresh_feed_l",
    "returned_to_balance_tank_l",
    "external_volume_balance_error_l",
    "unsafe_forward_l",
    "thermal_unsafe_forward_l",
    "pressure_noncompliant_forward_l",
    "contamination_exposed_forward_l",
    "unsafe_forward_fraction_of_forward",
    "quality_out_of_spec_l",
    "first_forward_time_s",
    "minimum_forward_holding_temp_c",
    "minimum_forward_actual_residence_time_s",
    "minimum_forward_relative_lethality",
    "minimum_forward_differential_pressure_bar",
    "maximum_forward_product_temp_c",
    "maximum_fouling_index",
    "final_fouling_index",
    "fouling_reduction_fraction",
    "final_cip_total_soil_g",
    "final_cip_residual_chemical_fraction",
    "cip_cleaning_complete",
    "thermal_energy_kwh",
    "pump_energy_kwh",
    "total_energy_kwh",
    "specific_energy_kwh_per_1000l_forward",
    "maximum_abs_mass_balance_error_l",
    "maximum_holding_inventory_error_l",
    "maximum_fdv_line_inventory_error_l",
    "maximum_post_fdv_inventory_error_l",
    "maximum_abs_post_fdv_balance_error_l",
    "maximum_abs_balance_tank_volume_error_l",
    "maximum_abs_balance_tank_temperature_moment_error_l_c",
    "maximum_abs_shadow_hx_energy_balance_error_kj",
    "alarm_activation_count",
)

FAULT_WINDOW_METRICS = (
    "fault_window_forward_l",
    "fault_window_diverted_l",
    "fault_window_unsafe_forward_l",
    "fault_window_quality_out_of_spec_l",
    "fault_window_alarm_duration_s",
    "fault_window_max_alarm_count",
    "first_fault_alarm_latency_s",
)

ANALYSIS_METRICS = SUMMARY_METRICS + FAULT_WINDOW_METRICS
COUNTERFACTUAL_METRICS = (
    "forward_l",
    "diverted_l",
    "unsafe_forward_l",
    "quality_out_of_spec_l",
    "total_energy_kwh",
)


def counterfactual_reference_scenario(scenario: str) -> str:
    """Return a mode-matched reference for conditional scenario attribution."""

    if scenario == "incomplete_cleaning":
        return "cip_cycle"
    return "normal"


OBSERVABLE_ALARM_COLUMNS = tuple(
    name for name in ALARM_COLUMNS if name != "alarm_unsafe_forward"
)


@dataclass(frozen=True)
class UniformRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("sampling bounds must be finite")
        if self.low >= self.high:
            raise ValueError(
                f"sampling range requires LOW < HIGH, got {self.low} >= {self.high}"
            )

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


def _derive_seed(master_seed: int, namespace: str, index: int) -> int:
    """Derive an order-independent 31-bit seed from stable text input."""
    payload = f"{master_seed}:{namespace}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _base_config(duration_s: float, dt_s: float, master_seed: int) -> HTSTConfig:
    # Preserve the public-reference defaults at 900 s while scaling event times
    # for short smoke or exploratory runs.
    default = HTSTConfig()
    fault_start_s = min(default.fault_start_s, 0.45 * duration_s)
    fault_duration_s = min(default.fault_duration_s, 0.20 * duration_s)
    return replace(
        default,
        duration_s=duration_s,
        dt_s=dt_s,
        random_seed=_derive_seed(master_seed, "baseline-model", 0),
        fault_start_s=fault_start_s,
        fault_duration_s=fault_duration_s,
    )


def default_sampling_ranges(config: HTSTConfig) -> dict[str, UniformRange]:
    """Return documented uniform uncertainty ranges around nominal values."""
    start = config.fault_start_s
    fault_duration = config.fault_duration_s
    return {
        "fault_start_s": UniformRange(max(0.0, 0.80 * start), 1.20 * start),
        "fault_duration_s": UniformRange(
            max(config.dt_s, 0.50 * fault_duration),
            max(config.dt_s * 1.01, 1.50 * fault_duration),
        ),
        "fault_severity": UniformRange(0.50, 1.20),
        "nominal_flow_l_h": UniformRange(
            0.90 * config.nominal_flow_l_h,
            1.10 * config.nominal_flow_l_h,
        ),
        "raw_milk_temp_c": UniformRange(
            config.raw_milk_temp_c - 2.0,
            config.raw_milk_temp_c + 4.0,
        ),
        "pasteurization_setpoint_c": UniformRange(
            max(config.diversion_threshold_c + 0.50, config.pasteurization_setpoint_c - 1.0),
            config.pasteurization_setpoint_c + 2.0,
        ),
        "forward_temperature_margin_c": UniformRange(0.20, 0.60),
        "nominal_holding_time_s": UniformRange(
            max(config.minimum_holding_time_s + 0.50, config.nominal_holding_time_s - 2.0),
            config.nominal_holding_time_s + 2.0,
        ),
        "regenerator_effectiveness": UniformRange(0.84, 0.95),
        "heater_tau_s": UniformRange(1.50, 3.00),
        "heater_max_delta_c": UniformRange(75.0, 90.0),
        "cooler_effectiveness": UniformRange(0.90, 0.98),
        "booster_pressure_gain_bar": UniformRange(1.00, 1.40),
        "fouling_rate_per_h": UniformRange(0.02, 0.08),
        "safety_sensor_noise_std_c": UniformRange(0.02, 0.10),
        "flow_meter_noise_std_fraction": UniformRange(0.001, 0.005),
        "sensor_to_fdv_delay_s": UniformRange(1.00, 2.00),
        "balance_tank_initial_volume_l": UniformRange(400.0, 800.0),
        "balance_tank_heat_loss_tau_s": UniformRange(3_600.0, 14_400.0),
        "post_fdv_residence_time_s": UniformRange(2.00, 10.00),
        "stationary_cooling_tau_s": UniformRange(1_800.0, 7_200.0),
        "fastest_flow_efficiency": UniformRange(0.80, 0.90),
        "fdv_travel_time_s": UniformRange(0.10, 0.80),
        "cip_caustic_removal_rate_s": UniformRange(0.004, 0.012),
        "cip_acid_removal_rate_s": UniformRange(0.002, 0.008),
        "cip_reference_velocity_m_s": UniformRange(1.20, 1.80),
        "cip_line_hold_up_l": UniformRange(80.0, 150.0),
    }


def _parse_range_overrides(
    raw_overrides: Sequence[Sequence[str]] | None,
    defaults: dict[str, UniformRange],
) -> dict[str, UniformRange]:
    ranges = dict(defaults)
    for raw in raw_overrides or ():
        name, low_text, high_text = raw
        if name not in defaults:
            choices = ", ".join(defaults)
            raise ValueError(f"unsupported range parameter {name!r}; choose from: {choices}")
        try:
            low = float(low_text)
            high = float(high_text)
        except ValueError as exc:
            raise ValueError(f"range for {name} must contain numeric LOW HIGH") from exc
        ranges[name] = UniformRange(low, high)
    return ranges


def _validate_ranges(config: HTSTConfig, ranges: dict[str, UniformRange]) -> None:
    """Exercise endpoints and pairwise joint corners through config validation."""
    config_fields = asdict(config)
    for name, bounds in ranges.items():
        if name not in config_fields:
            raise ValueError(f"HTSTConfig has no field {name!r}")
        for value in (bounds.low, bounds.high):
            try:
                replace(config, **{name: value})
            except ValueError as exc:
                raise ValueError(f"invalid {name} endpoint {value}: {exc}") from exc
    central_values = {name: bounds.midpoint for name, bounds in ranges.items()}
    try:
        replace(config, **central_values)
    except ValueError as exc:
        raise ValueError(f"invalid joint midpoint configuration: {exc}") from exc

    for name, bounds in ranges.items():
        for value in (bounds.low, bounds.high):
            joint_values = dict(central_values)
            joint_values[name] = value
            try:
                replace(config, **joint_values)
            except ValueError as exc:
                raise ValueError(
                    f"invalid joint endpoint {name}={value} against range midpoints: {exc}"
                ) from exc

    # Relational constraints can be satisfied by each endpoint against the base
    # configuration while failing when two sampled parameters meet at a corner.
    # Pairwise corners cover the current HTSTConfig relations without requiring
    # an exponential enumeration of every range combination.
    for first, second in combinations(ranges, 2):
        for first_value, second_value in product(
            (ranges[first].low, ranges[first].high),
            (ranges[second].low, ranges[second].high),
        ):
            values = dict(central_values)
            values[first] = first_value
            values[second] = second_value
            try:
                replace(config, **values)
            except ValueError as exc:
                raise ValueError(
                    "invalid joint range corner "
                    f"{first}={first_value}, {second}={second_value}: {exc}"
                ) from exc


def sample_config(
    base: HTSTConfig,
    ranges: dict[str, UniformRange],
    master_seed: int,
    sample_index: int,
) -> tuple[HTSTConfig, int]:
    """Sample one joint configuration, independent of scenario ordering."""
    sampling_seed = _derive_seed(master_seed, "parameter-sampling", sample_index)
    rng = random.Random(sampling_seed)
    values = {name: bounds.sample(rng) for name, bounds in ranges.items()}
    values["random_seed"] = _derive_seed(master_seed, "model-noise", sample_index)
    try:
        config = replace(base, **values)
    except ValueError as exc:
        sampled = ", ".join(
            f"{name}={values[name]!r}" for name in sorted(values)
        )
        raise ValueError(
            "invalid sampled configuration at "
            f"sample_index={sample_index}, sampling_seed={sampling_seed}: {exc}; "
            f"sampled values: {sampled}"
        ) from exc
    return config, sampling_seed


def _number_or_none(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite metric encountered: {value}")
        return value
    raise TypeError(f"metric must be numeric or None, got {type(value).__name__}")


def _fault_window_metrics(
    records: list[dict[str, float | int | str]],
    config: HTSTConfig,
) -> dict[str, int | float | None]:
    active = [row for row in records if int(row["fault_active"]) == 1]
    previous = {name: 0 for name in OBSERVABLE_ALARM_COLUMNS}
    first_alarm_time: float | None = None
    for row in records:
        current = {name: int(row[name]) for name in OBSERVABLE_ALARM_COLUMNS}
        if int(row["fault_active"]) == 1 and first_alarm_time is None:
            newly_active = any(
                current[name] and not previous[name]
                for name in OBSERVABLE_ALARM_COLUMNS
            )
            if newly_active:
                first_alarm_time = float(row["time_start_s"])
        previous = current

    def observable_count(row: dict[str, float | int | str]) -> int:
        return sum(int(row[name]) for name in OBSERVABLE_ALARM_COLUMNS)

    return {
        "fault_window_forward_l": sum(float(row["forward_l"]) for row in active),
        "fault_window_diverted_l": sum(float(row["diverted_l"]) for row in active),
        "fault_window_unsafe_forward_l": sum(
            float(row["unsafe_forward_l"]) for row in active
        ),
        "fault_window_quality_out_of_spec_l": sum(
            float(row["quality_out_of_spec_l"]) for row in active
        ),
        "fault_window_alarm_duration_s": sum(
            float(row["step_dt_s"])
            for row in active
            if observable_count(row) > 0
        ),
        "fault_window_max_alarm_count": max(
            (observable_count(row) for row in active),
            default=0,
        ),
        "first_fault_alarm_latency_s": (
            max(0.0, first_alarm_time - config.fault_start_s)
            if first_alarm_time is not None
            else None
        ),
    }


def _observable_alarm_activation_count(
    records: Sequence[dict[str, float | int | str]],
) -> int:
    previous = {name: 0 for name in OBSERVABLE_ALARM_COLUMNS}
    activations = 0
    for row in records:
        for name in OBSERVABLE_ALARM_COLUMNS:
            state = int(row[name])
            if state and not previous[name]:
                activations += 1
            previous[name] = state
    return activations


def evaluate_run(
    scenario: str,
    config: HTSTConfig,
) -> tuple[dict[str, object], dict[str, int | float | None]]:
    records = HTSTSimulator(config).run(scenario)
    run_summary = summarize(records, config)
    metrics = {
        name: _number_or_none(run_summary[name])
        for name in SUMMARY_METRICS
    }
    metrics["alarm_activation_count"] = _observable_alarm_activation_count(records)
    metrics.update(_fault_window_metrics(records, config))
    return run_summary, metrics


def run_monte_carlo(
    scenarios: Sequence[str],
    runs_per_scenario: int,
    base: HTSTConfig,
    ranges: dict[str, UniformRange],
    master_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Build the complete plan first.  A bad joint draw therefore fails with its
    # exact index/seed/values before any expensive scenario simulation starts.
    sampling_plan = [
        sample_config(
            base,
            ranges,
            master_seed,
            sample_index,
        )
        for sample_index in range(runs_per_scenario)
    ]
    reference_scenarios = {
        counterfactual_reference_scenario(scenario) for scenario in scenarios
    }
    for sample_index, (config, sampling_seed) in enumerate(sampling_plan):
        sampled_values = {name: getattr(config, name) for name in ranges}
        reference_runs = {
            reference: evaluate_run(reference, config)
            for reference in reference_scenarios
        }
        for scenario in scenarios:
            if scenario in reference_runs:
                run_summary, metrics = reference_runs[scenario]
            else:
                run_summary, metrics = evaluate_run(scenario, config)
            reference_scenario = counterfactual_reference_scenario(scenario)
            reference_summary, reference_metrics = reference_runs[reference_scenario]
            counterfactual_values: dict[str, object] = {
                "counterfactual_reference_scenario": reference_scenario,
                "counterfactual_reference_run_id": reference_summary["run_id"],
                "counterfactual_reference_config_hash": reference_summary["config_hash"],
            }
            for metric in COUNTERFACTUAL_METRICS:
                reference_value = reference_metrics[metric]
                scenario_value = metrics[metric]
                counterfactual_values[f"counterfactual_{metric}"] = reference_value
                counterfactual_values[f"delta_vs_counterfactual_{metric}"] = (
                    None
                    if reference_value is None or scenario_value is None
                    else float(scenario_value) - float(reference_value)
                )
            rows.append(
                {
                    "sample_id": f"mc-{sample_index:06d}",
                    "sample_index": sample_index,
                    "scenario": scenario,
                    "sampling_seed": sampling_seed,
                    "random_seed": config.random_seed,
                    "model_version": run_summary["model_version"],
                    "run_id": run_summary["run_id"],
                    "config_hash": run_summary["config_hash"],
                    **sampled_values,
                    **metrics,
                    **counterfactual_values,
                }
            )
    return rows


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _describe(values: Iterable[object]) -> dict[str, int | float | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "sample_stddev": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "sample_stddev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": clean[0],
        "p05": _percentile(clean, 0.05),
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "max": clean[-1],
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def aggregate_monte_carlo(
    rows: Sequence[dict[str, object]],
    scenarios: Sequence[str],
    runs_per_scenario: int,
    ranges: dict[str, UniformRange],
    base: HTSTConfig,
    master_seed: int,
) -> dict[str, object]:
    by_scenario: dict[str, object] = {}
    for scenario in scenarios:
        selected = [row for row in rows if row["scenario"] == scenario]
        metric_summary = {
            metric: {
                **_describe(row[metric] for row in selected),
                "missing_count": sum(row[metric] is None for row in selected),
            }
            for metric in ANALYSIS_METRICS
        }
        incidence: dict[str, object] = {}
        for label, metric in (
            ("unsafe_forward", "unsafe_forward_l"),
            ("fault_window_unsafe_forward", "fault_window_unsafe_forward_l"),
            ("quality_out_of_spec", "quality_out_of_spec_l"),
            ("fault_window_quality_out_of_spec", "fault_window_quality_out_of_spec_l"),
        ):
            positives = sum(float(row[metric] or 0.0) > 1e-12 for row in selected)
            low, high = _wilson_interval(positives, len(selected))
            incidence[label] = {
                "positive_runs": positives,
                "total_runs": len(selected),
                "probability": positives / len(selected) if selected else None,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        by_scenario[scenario] = {
            "metrics": metric_summary,
            "incidence": incidence,
            "counterfactual_reference_scenario": counterfactual_reference_scenario(
                scenario
            ),
            "counterfactual_deltas": {
                metric: {
                    **_describe(
                        row[f"delta_vs_counterfactual_{metric}"] for row in selected
                    ),
                    "missing_count": sum(
                        row[f"delta_vs_counterfactual_{metric}"] is None
                        for row in selected
                    ),
                }
                for metric in COUNTERFACTUAL_METRICS
            },
        }
    reference_scenarios = {
        counterfactual_reference_scenario(scenario) for scenario in scenarios
    }
    additional_reference_evaluations = runs_per_scenario * len(
        reference_scenarios - set(scenarios)
    )
    return {
        "analysis": "monte_carlo",
        "model_version": MODEL_VERSION,
        "master_seed": master_seed,
        "sampling_design": (
            "independent uniform parameters with paired samples across scenarios; "
            "each sample also has a same-config/same-noise mode-matched "
            "counterfactual (normal for production, cip_cycle for incomplete cleaning)"
        ),
        "model_seed_design": "SHA-256-derived 31-bit seed per sample",
        "fault_window_definition": (
            "records whose configured fault_active field equals 1; total-run metrics "
            "also capture scenario effects that persist after this window"
        ),
        "runs_per_scenario": runs_per_scenario,
        "scenarios": list(scenarios),
        "scenario_simulations": len(rows),
        "counterfactual_reference_scenarios": sorted(reference_scenarios),
        "additional_counterfactual_simulations": additional_reference_evaluations,
        "total_simulations": len(rows) + additional_reference_evaluations,
        "counterfactual_metrics": list(COUNTERFACTUAL_METRICS),
        "base_config": asdict(base),
        "sampling_ranges": {
            name: {
                "distribution": "uniform",
                "low": bounds.low,
                "high": bounds.high,
            }
            for name, bounds in ranges.items()
        },
        "random_seed_distribution": {
            "method": "deterministic hash derivation",
            "low_inclusive": 0,
            "high_exclusive": 2**31,
        },
        "scenario_results": by_scenario,
    }


def _central_config(
    base: HTSTConfig,
    ranges: dict[str, UniformRange],
    master_seed: int,
) -> HTSTConfig:
    values = {name: bounds.midpoint for name, bounds in ranges.items()}
    values["random_seed"] = _derive_seed(master_seed, "sensitivity-model", 0)
    return replace(base, **values)


def run_oat_sensitivity(
    scenarios: Sequence[str],
    base: HTSTConfig,
    ranges: dict[str, UniformRange],
    master_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    central = _central_config(base, ranges, master_seed)
    rows: list[dict[str, object]] = []
    baseline_metrics_by_scenario: dict[str, dict[str, int | float | None]] = {}
    baseline_run_ids: dict[str, str] = {}

    for scenario in scenarios:
        baseline_summary, baseline_metrics = evaluate_run(scenario, central)
        baseline_metrics_by_scenario[scenario] = baseline_metrics
        baseline_run_ids[scenario] = str(baseline_summary["run_id"])
        for parameter, bounds in ranges.items():
            low_config = replace(central, **{parameter: bounds.low})
            high_config = replace(central, **{parameter: bounds.high})
            _, low_metrics = evaluate_run(scenario, low_config)
            _, high_metrics = evaluate_run(scenario, high_config)
            input_span = bounds.high - bounds.low
            for metric in ANALYSIS_METRICS:
                low_value = low_metrics[metric]
                baseline_value = baseline_metrics[metric]
                high_value = high_metrics[metric]
                complete = (
                    low_value is not None
                    and baseline_value is not None
                    and high_value is not None
                )
                if complete:
                    low_number = float(low_value)
                    baseline_number = float(baseline_value)
                    high_number = float(high_value)
                    absolute_change = high_number - low_number
                    slope = absolute_change / input_span
                    output_scale = max(
                        abs(low_number),
                        abs(baseline_number),
                        abs(high_number),
                    )
                    scaled_change = (
                        absolute_change / output_scale if output_scale > 1e-15 else 0.0
                    )
                    elasticity = (
                        slope * bounds.midpoint / baseline_number
                        if abs(baseline_number) > 1e-15
                        else None
                    )
                else:
                    absolute_change = None
                    slope = None
                    scaled_change = None
                    elasticity = None
                rows.append(
                    {
                        "scenario": scenario,
                        "parameter": parameter,
                        "metric": metric,
                        "low_input": bounds.low,
                        "baseline_input": bounds.midpoint,
                        "high_input": bounds.high,
                        "low_metric": low_value,
                        "baseline_metric": baseline_value,
                        "high_metric": high_value,
                        "absolute_change_high_minus_low": absolute_change,
                        "slope_per_input_unit": slope,
                        "scaled_output_change": scaled_change,
                        "elasticity_at_baseline": elasticity,
                        "absolute_effect_rank": None,
                    }
                )

    # Rank parameters independently for each scenario/output.  The robust
    # scaled change is used so zero-baseline safety metrics remain rankable.
    for scenario in scenarios:
        for metric in ANALYSIS_METRICS:
            candidates = [
                row
                for row in rows
                if row["scenario"] == scenario
                and row["metric"] == metric
                and row["scaled_output_change"] is not None
                and abs(float(row["scaled_output_change"])) > 1e-15
            ]
            candidates.sort(
                key=lambda row: (
                    -abs(float(row["scaled_output_change"])),
                    str(row["parameter"]),
                )
            )
            for rank, row in enumerate(candidates, start=1):
                row["absolute_effect_rank"] = rank

    rankings: dict[str, dict[str, list[dict[str, object]]]] = {}
    for scenario in scenarios:
        rankings[scenario] = {}
        for metric in ANALYSIS_METRICS:
            selected = sorted(
                (
                    row for row in rows
                    if row["scenario"] == scenario
                    and row["metric"] == metric
                    and row["absolute_effect_rank"] is not None
                ),
                key=lambda row: (
                    row["absolute_effect_rank"] is None,
                    row["absolute_effect_rank"] or math.inf,
                ),
            )
            rankings[scenario][metric] = [
                {
                    "rank": row["absolute_effect_rank"],
                    "parameter": row["parameter"],
                    "scaled_output_change": row["scaled_output_change"],
                    "elasticity_at_baseline": row["elasticity_at_baseline"],
                }
                for row in selected
            ]

    report = {
        "analysis": "one_at_a_time_sensitivity",
        "model_version": MODEL_VERSION,
        "master_seed": master_seed,
        "method": (
            "one parameter at uniform-range endpoints; all other parameters at "
            "midpoints; common model-noise seed"
        ),
        "fault_window_definition": (
            "records whose configured fault_active field equals 1; total-run metrics "
            "also capture scenario effects that persist after this window"
        ),
        "central_config": asdict(central),
        "baseline_run_ids": baseline_run_ids,
        "simulation_count": len(scenarios) * (1 + 2 * len(ranges)),
        "metrics": list(ANALYSIS_METRICS),
        "parameter_ranges": {
            name: {"low": bounds.low, "midpoint": bounds.midpoint, "high": bounds.high}
            for name, bounds in ranges.items()
        },
        "baseline_metrics": baseline_metrics_by_scenario,
        "rankings": rankings,
        "rows": rows,
    }
    return rows, report


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_NAMES,
        default=list(DEFAULT_SCENARIOS),
        help="scenarios to evaluate (default: representative production faults)",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=100,
        help="Monte Carlo samples per scenario (default: 100)",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260724,
        help="master seed controlling parameters and model-noise seeds",
    )
    parser.add_argument("--duration", type=_finite_float, default=900.0, help="seconds")
    parser.add_argument("--dt", type=_finite_float, default=0.5, help="seconds")
    parser.add_argument(
        "--range",
        dest="range_overrides",
        action="append",
        nargs=3,
        metavar=("PARAMETER", "LOW", "HIGH"),
        help="override a uniform sampling range; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base = _base_config(args.duration, args.dt, args.master_seed)
        ranges = _parse_range_overrides(
            args.range_overrides,
            default_sampling_ranges(base),
        )
        _validate_ranges(base, ranges)
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    try:
        monte_carlo_rows = run_monte_carlo(
            args.scenarios,
            args.runs,
            base,
            ranges,
            args.master_seed,
        )
        monte_carlo_summary = aggregate_monte_carlo(
            monte_carlo_rows,
            args.scenarios,
            args.runs,
            ranges,
            base,
            args.master_seed,
        )
        sensitivity_rows, sensitivity_report = run_oat_sensitivity(
            args.scenarios,
            base,
            ranges,
            args.master_seed,
        )
    except ValueError as exc:
        raise SystemExit(f"analysis error: {exc}") from exc

    output = args.output.expanduser().resolve()
    source = source_provenance(
        {
            "model.py": HERE / "model.py",
            "process_components.py": HERE / "process_components.py",
            "provenance.py": HERE / "provenance.py",
            "uncertainty.py": HERE / "uncertainty.py",
        }
    )
    reproduce_parts = [
        "python3",
        "uncertainty.py",
        "--runs",
        str(args.runs),
        "--duration",
        str(args.duration),
        "--dt",
        str(args.dt),
        "--master-seed",
        str(args.master_seed),
        "--scenarios",
        *args.scenarios,
    ]
    for override in args.range_overrides or ():
        reproduce_parts.extend(("--range", *override))
    with staged_output_directory(output) as staging:
        runs_path = staging / "monte_carlo_runs.csv"
        summary_path = staging / "monte_carlo_summary.json"
        sensitivity_csv_path = staging / "sensitivity.csv"
        sensitivity_json_path = staging / "sensitivity.json"
        _write_csv(runs_path, monte_carlo_rows)
        _write_json(summary_path, monte_carlo_summary)
        _write_csv(sensitivity_csv_path, sensitivity_rows)
        _write_json(sensitivity_json_path, sensitivity_report)
        artifacts = [
            runs_path,
            summary_path,
            sensitivity_csv_path,
            sensitivity_json_path,
        ]
        manifest_path = staging / "uncertainty_manifest.json"
        _write_json(
            manifest_path,
            {
                "analysis": "monte_carlo_and_one_at_a_time_sensitivity",
                "model_status": "unvalidated engineering research surrogate",
                "model_version": MODEL_VERSION,
                "python_version": platform.python_version(),
                "master_seed": args.master_seed,
                "runs_per_scenario": args.runs,
                "scenarios": list(args.scenarios),
                "monte_carlo_scenario_simulations": monte_carlo_summary[
                    "scenario_simulations"
                ],
                "monte_carlo_additional_counterfactual_simulations": monte_carlo_summary[
                    "additional_counterfactual_simulations"
                ],
                "monte_carlo_total_simulations": monte_carlo_summary[
                    "total_simulations"
                ],
                "oat_simulations": sensitivity_report["simulation_count"],
                "base_config": asdict(base),
                "sampling_ranges": {
                    name: {
                        "distribution": "uniform",
                        "low": bounds.low,
                        "high": bounds.high,
                    }
                    for name, bounds in ranges.items()
                },
                "source_provenance": source,
                "artifacts": [path.name for path in artifacts],
                "reproduce": " ".join(reproduce_parts),
            },
        )
        artifacts.append(manifest_path)
        write_checksums(staging, artifacts)

    print(
        "Monte Carlo simulations: "
        f"{monte_carlo_summary['scenario_simulations']} scenario + "
        f"{monte_carlo_summary['additional_counterfactual_simulations']} "
        "mode-matched counterfactual = "
        f"{monte_carlo_summary['total_simulations']}"
    )
    print(f"OAT simulations: {sensitivity_report['simulation_count']}")
    print(f"Runs CSV: {output / 'monte_carlo_runs.csv'}")
    print(f"Summary JSON: {output / 'monte_carlo_summary.json'}")
    print(f"Sensitivity CSV: {output / 'sensitivity.csv'}")
    print(f"Sensitivity JSON: {output / 'sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
