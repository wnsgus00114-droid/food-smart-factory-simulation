#!/usr/bin/env python3
"""Research-grade dynamic surrogate of a continuous HTST milk process.

Physical truth, observable instrumentation, controller state and actuator
state are separated.  The implementation is deterministic for a fixed seed,
conserves transported volume, handles exact event boundaries, and integrates
safe/unsafe outlet volume at parcel level.  It is not a regulatory or
organism-specific validation tool.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from typing import Any, Deque, Iterable, Mapping, Sequence

try:  # Support both package imports and direct script execution.
    from .process_components import (
        BalanceTank,
        BalanceTankConfig,
        CIPSoilConfig,
        CIPSoilModel,
        CIPStep,
        DynamicHeatExchanger,
        HeatExchangerConfig,
        LiquidStream,
    )
except ImportError:  # pragma: no cover - exercised by CLI integration tests
    from process_components import (
        BalanceTank,
        BalanceTankConfig,
        CIPSoilConfig,
        CIPSoilModel,
        CIPStep,
        DynamicHeatExchanger,
        HeatExchangerConfig,
        LiquidStream,
    )


MODEL_VERSION = "2.2.0"
STATE_SCHEMA_VERSION = "1.0.0"

MINIMUM_SUPPORTED_DT_S = 1.0e-3
MAXIMUM_SUPPORTED_DT_S = 1.0
DEFAULT_MAXIMUM_STEPS = 1_000_000
_VOLUME_EPSILON_L = 1.0e-10
_ABSOLUTE_ZERO_C = -273.15
_MAX_MODEL_TEMPERATURE_C = 1_000.0


@dataclass(frozen=True)
class HTSTConfig:
    # Numerical experiment
    dt_s: float = 0.5
    duration_s: float = 900.0
    random_seed: int = 20260724
    fault_start_s: float = 300.0
    fault_duration_s: float = 120.0
    fault_severity: float = 1.0
    maximum_steps: int = DEFAULT_MAXIMUM_STEPS

    # Public-reference operating envelope
    nominal_flow_l_h: float = 20_000.0
    raw_milk_temp_c: float = 4.0
    pasteurization_setpoint_c: float = 74.0
    diversion_threshold_c: float = 72.0
    forward_temperature_margin_c: float = 0.30
    nominal_holding_time_s: float = 18.0
    minimum_holding_time_s: float = 15.0
    final_product_target_c: float = 4.0
    maximum_product_temp_c: float = 6.0
    ambient_temp_c: float = 20.0

    # Product properties: provisional engineering assumptions
    density_kg_l: float = 1.03
    heat_capacity_kj_kg_k: float = 3.90

    # Balance tank and return circuit
    balance_tank_capacity_l: float = 1_200.0
    balance_tank_initial_volume_l: float = 600.0
    balance_tank_minimum_volume_l: float = 50.0
    balance_tank_heat_loss_tau_s: float = 7_200.0

    # Conservative detailed heat-exchanger shadow network.  The established
    # effectiveness model remains the controller plant; these states expose a
    # separately auditable energy-conserving thermal network for experiments.
    hx_regenerator_clean_ua_kw_k: float = 40.0
    hx_heater_clean_ua_kw_k: float = 50.0
    hx_cooler_clean_ua_kw_k: float = 45.0
    hx_product_holdup_l: float = 25.0
    hx_utility_holdup_l: float = 25.0
    hx_wall_heat_capacity_kj_k: float = 60.0
    hx_ambient_ua_kw_k: float = 0.02
    hx_fouling_resistance_ratio_at_one: float = 4.0
    hx_internal_max_dt_s: float = 0.05
    heating_utility_temp_c: float = 120.0
    cooling_utility_temp_c: float = 1.0

    # Thermal equipment
    regenerator_effectiveness: float = 0.90
    regenerator_tau_s: float = 4.0
    heater_tau_s: float = 2.0
    heater_max_delta_c: float = 82.0
    cooler_effectiveness: float = 0.98
    stationary_cooling_tau_s: float = 3_600.0
    post_fdv_residence_time_s: float = 5.0

    # Instrumentation, valve and line delays
    preheat_sensor_noise_std_c: float = 0.03
    control_sensor_noise_std_c: float = 0.05
    safety_sensor_noise_std_c: float = 0.05
    sensor_disagreement_limit_c: float = 0.75
    flow_meter_noise_std_fraction: float = 0.002
    pressure_sensor_noise_std_bar: float = 0.01
    booster_speed_sensor_noise_std_fraction: float = 0.003
    leak_detector_noise_std_fraction: float = 0.0005
    leak_detector_alarm_fraction: float = 0.005
    product_sensor_noise_std_c: float = 0.05
    required_pressure_differential_bar: float = 0.50
    forward_confirmation_s: float = 1.0
    divert_actuation_delay_s: float = 0.20
    sensor_to_fdv_delay_s: float = 1.50
    fdv_travel_time_s: float = 0.20
    fdv_feedback_tolerance: float = 0.05
    fdv_position_sensor_noise_std_fraction: float = 0.002

    # Hydraulic surrogate
    raw_side_pressure_bar: float = 2.00
    booster_pressure_gain_bar: float = 1.20
    pasteurized_line_drop_bar: float = 0.25
    fouling_pressure_drop_bar: float = 0.35
    pump_efficiency: float = 0.68

    # PI controller
    controller_kp: float = 0.005
    controller_ki: float = 0.0002

    # Fouling and CIP surrogate
    initial_fouling_index: float = 0.0
    fouling_rate_per_h: float = 0.04
    fouling_heat_loss_fraction: float = 0.45
    fouling_regeneration_loss_fraction: float = 0.25
    fouling_alarm_index: float = 0.55
    cip_initial_fouling_index: float = 0.65
    cip_caustic_removal_rate_s: float = 0.008
    cip_acid_removal_rate_s: float = 0.004
    cip_initial_protein_soil_g: float = 650.0
    cip_initial_mineral_soil_g: float = 350.0
    cip_reference_velocity_m_s: float = 1.5
    cip_line_hold_up_l: float = 100.0
    cip_clean_soil_threshold_g: float = 300.0
    cip_residual_chemical_threshold_fraction: float = 0.002

    # Relative thermal-treatment diagnostic; not organism-specific
    lethality_reference_temp_c: float = 72.0
    lethality_reference_time_s: float = 15.0
    lethality_z_c: float = 7.0
    fastest_flow_efficiency: float = 0.85
    max_exponential_argument: float = 80.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real numeric value")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}")
        strictly_positive = (
            "dt_s",
            "duration_s",
            "nominal_flow_l_h",
            "nominal_holding_time_s",
            "minimum_holding_time_s",
            "density_kg_l",
            "heat_capacity_kj_kg_k",
            "balance_tank_capacity_l",
            "balance_tank_heat_loss_tau_s",
            "hx_regenerator_clean_ua_kw_k",
            "hx_heater_clean_ua_kw_k",
            "hx_cooler_clean_ua_kw_k",
            "hx_product_holdup_l",
            "hx_utility_holdup_l",
            "hx_wall_heat_capacity_kj_k",
            "hx_internal_max_dt_s",
            "regenerator_tau_s",
            "heater_tau_s",
            "heater_max_delta_c",
            "stationary_cooling_tau_s",
            "post_fdv_residence_time_s",
            "lethality_reference_time_s",
            "lethality_z_c",
            "pump_efficiency",
            "required_pressure_differential_bar",
            "raw_side_pressure_bar",
            "booster_pressure_gain_bar",
            "fdv_travel_time_s",
            "fastest_flow_efficiency",
            "max_exponential_argument",
            "cip_initial_protein_soil_g",
            "cip_initial_mineral_soil_g",
            "cip_reference_velocity_m_s",
            "cip_line_hold_up_l",
            "cip_clean_soil_threshold_g",
        )
        for name in strictly_positive:
            if float(values[name]) <= 0.0:
                raise ValueError(f"{name} must be positive, got {values[name]}")
        if self.dt_s < MINIMUM_SUPPORTED_DT_S:
            raise ValueError(
                f"dt_s must be >= {MINIMUM_SUPPORTED_DT_S:g} s to bound step count"
            )
        if self.dt_s > MAXIMUM_SUPPORTED_DT_S:
            raise ValueError(
                f"dt_s must be <= {MAXIMUM_SUPPORTED_DT_S:g} s for valve and transport resolution"
            )
        if isinstance(self.maximum_steps, bool) or not isinstance(self.maximum_steps, int):
            raise ValueError("maximum_steps must be an integer")
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        estimated_step_ratio = self.duration_s / self.dt_s
        if (
            not math.isfinite(estimated_step_ratio)
            or estimated_step_ratio > self.maximum_steps
        ):
            raise ValueError(
                "duration_s / dt_s exceeds maximum_steps "
                f"({estimated_step_ratio!r} > {self.maximum_steps})"
            )
        estimated_steps = math.ceil(estimated_step_ratio)
        if self.minimum_holding_time_s > self.nominal_holding_time_s:
            raise ValueError("minimum holding time cannot exceed nominal holding time")
        if self.diversion_threshold_c >= self.pasteurization_setpoint_c:
            raise ValueError("pasteurization setpoint must exceed diversion threshold")
        bounded_zero_one = (
            "regenerator_effectiveness",
            "cooler_effectiveness",
            "pump_efficiency",
            "initial_fouling_index",
            "cip_initial_fouling_index",
            "fouling_heat_loss_fraction",
            "fouling_regeneration_loss_fraction",
            "fouling_alarm_index",
            "fastest_flow_efficiency",
            "fdv_feedback_tolerance",
        )
        for name in bounded_zero_one:
            if not 0.0 <= float(values[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        nonnegative = (
            "fault_start_s",
            "fault_duration_s",
            "fault_severity",
            "forward_temperature_margin_c",
            "preheat_sensor_noise_std_c",
            "control_sensor_noise_std_c",
            "safety_sensor_noise_std_c",
            "flow_meter_noise_std_fraction",
            "pressure_sensor_noise_std_bar",
            "booster_speed_sensor_noise_std_fraction",
            "leak_detector_noise_std_fraction",
            "leak_detector_alarm_fraction",
            "product_sensor_noise_std_c",
            "fdv_position_sensor_noise_std_fraction",
            "forward_confirmation_s",
            "divert_actuation_delay_s",
            "sensor_to_fdv_delay_s",
            "sensor_disagreement_limit_c",
            "pasteurized_line_drop_bar",
            "fouling_pressure_drop_bar",
            "controller_kp",
            "controller_ki",
            "fouling_rate_per_h",
            "cip_caustic_removal_rate_s",
            "cip_acid_removal_rate_s",
            "balance_tank_initial_volume_l",
            "balance_tank_minimum_volume_l",
            "cip_residual_chemical_threshold_fraction",
            "hx_ambient_ua_kw_k",
            "hx_fouling_resistance_ratio_at_one",
        )
        for name in nonnegative:
            if float(values[name]) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.fault_severity > 2.0:
            raise ValueError("fault_severity must be <= 2.0")
        if self.balance_tank_initial_volume_l > self.balance_tank_capacity_l:
            raise ValueError("balance tank initial volume cannot exceed capacity")
        if self.balance_tank_minimum_volume_l > self.balance_tank_initial_volume_l:
            raise ValueError("balance tank minimum volume cannot exceed initial volume")
        if self.cip_residual_chemical_threshold_fraction > 1.0:
            raise ValueError("CIP residual chemical threshold must be <= 1")
        if self.cip_clean_soil_threshold_g > (
            self.cip_initial_protein_soil_g + self.cip_initial_mineral_soil_g
        ):
            raise ValueError("CIP clean-soil threshold cannot exceed initial soil")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.diversion_threshold_c + self.forward_temperature_margin_c >= self.pasteurization_setpoint_c:
            raise ValueError("forward threshold plus margin must remain below setpoint")
        if self.maximum_product_temp_c < self.final_product_target_c:
            raise ValueError("maximum product temperature cannot be below its target")
        if self.leak_detector_alarm_fraction > 1.0:
            raise ValueError("leak detector alarm fraction must be <= 1.0")
        temperature_fields = (
            "raw_milk_temp_c",
            "pasteurization_setpoint_c",
            "diversion_threshold_c",
            "final_product_target_c",
            "maximum_product_temp_c",
            "ambient_temp_c",
            "lethality_reference_temp_c",
            "heating_utility_temp_c",
            "cooling_utility_temp_c",
        )
        for name in temperature_fields:
            value = float(values[name])
            if not _ABSOLUTE_ZERO_C <= value <= _MAX_MODEL_TEMPERATURE_C:
                raise ValueError(
                    f"{name} must be in [{_ABSOLUTE_ZERO_C}, {_MAX_MODEL_TEMPERATURE_C}]"
                )
        if self.heater_max_delta_c > _MAX_MODEL_TEMPERATURE_C:
            raise ValueError("heater_max_delta_c exceeds the supported numeric range")
        if self.nominal_flow_l_h > 1.0e9:
            raise ValueError("nominal_flow_l_h exceeds the supported numeric range")
        upper_bounded = {
            "density_kg_l": 1.0e6,
            "heat_capacity_kj_kg_k": 1.0e6,
            "balance_tank_capacity_l": 1.0e9,
            "balance_tank_initial_volume_l": 1.0e9,
            "balance_tank_minimum_volume_l": 1.0e9,
            "balance_tank_heat_loss_tau_s": 1.0e12,
            "hx_regenerator_clean_ua_kw_k": 1.0e9,
            "hx_heater_clean_ua_kw_k": 1.0e9,
            "hx_cooler_clean_ua_kw_k": 1.0e9,
            "hx_product_holdup_l": 1.0e9,
            "hx_utility_holdup_l": 1.0e9,
            "hx_wall_heat_capacity_kj_k": 1.0e12,
            "hx_ambient_ua_kw_k": 1.0e9,
            "hx_fouling_resistance_ratio_at_one": 1.0e9,
            "hx_internal_max_dt_s": MAXIMUM_SUPPORTED_DT_S,
            "cip_initial_protein_soil_g": 1.0e12,
            "cip_initial_mineral_soil_g": 1.0e12,
            "cip_reference_velocity_m_s": 1.0e6,
            "cip_line_hold_up_l": 1.0e9,
            "cip_clean_soil_threshold_g": 1.0e12,
        }
        for name, upper in upper_bounded.items():
            if float(values[name]) > upper:
                raise ValueError(f"{name} exceeds the supported numeric range")
        for name in (
            "nominal_holding_time_s",
            "sensor_to_fdv_delay_s",
            "post_fdv_residence_time_s",
        ):
            queue_ratio = float(values[name]) / self.dt_s
            if not math.isfinite(queue_ratio) or queue_ratio > self.maximum_steps:
                raise ValueError(
                    f"{name} / dt_s exceeds maximum_steps "
                    f"({queue_ratio!r} > {self.maximum_steps})"
                )
        for name in ("controller_kp", "controller_ki"):
            if float(values[name]) > 1.0e3:
                raise ValueError(f"{name} exceeds the supported numeric range")
        for name in (
            "preheat_sensor_noise_std_c",
            "control_sensor_noise_std_c",
            "safety_sensor_noise_std_c",
            "product_sensor_noise_std_c",
        ):
            if float(values[name]) > _MAX_MODEL_TEMPERATURE_C:
                raise ValueError(f"{name} exceeds the supported numeric range")
        for name in (
            "flow_meter_noise_std_fraction",
            "booster_speed_sensor_noise_std_fraction",
            "leak_detector_noise_std_fraction",
            "fdv_position_sensor_noise_std_fraction",
        ):
            if float(values[name]) > 10.0:
                raise ValueError(f"{name} exceeds the supported numeric range")
        if self.max_exponential_argument > 700.0:
            raise ValueError("max_exponential_argument must be <= 700")
        if not math.isfinite(self.fault_start_s + self.fault_duration_s):
            raise ValueError("fault interval endpoint must be finite")
        for name, value in (
            ("holding_tube_volume_l", self.holding_tube_volume_l),
            ("fdv_line_volume_l", self.fdv_line_volume_l),
            ("post_fdv_inventory_l", self.post_fdv_inventory_l),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def nominal_flow_l_s(self) -> float:
        return self.nominal_flow_l_h / 3600.0

    @property
    def holding_tube_volume_l(self) -> float:
        return self.nominal_flow_l_s * self.nominal_holding_time_s

    @property
    def maximum_safe_flow_l_h(self) -> float:
        return (
            self.holding_tube_volume_l
            * self.fastest_flow_efficiency
            / self.minimum_holding_time_s
            * 3600.0
        )

    @property
    def fdv_line_volume_l(self) -> float:
        return self.nominal_flow_l_s * self.sensor_to_fdv_delay_s

    @property
    def post_fdv_inventory_l(self) -> float:
        return self.nominal_flow_l_s * self.post_fdv_residence_time_s


@dataclass(frozen=True)
class CampaignPhase:
    """A stage-local scenario executed on one continuous simulator state."""

    phase_id: str
    scenario: str
    duration_s: float

    def __post_init__(self) -> None:
        if not self.phase_id or not self.phase_id.strip():
            raise ValueError("phase_id must be non-empty")
        if self.scenario not in SCENARIO_NAMES:
            raise ValueError(f"unknown campaign scenario {self.scenario!r}")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("campaign phase duration_s must be finite and positive")


@dataclass
class FluidParcel:
    volume_l: float
    temp_c: float
    entry_start_s: float
    entry_end_s: float
    mean_pass_count: float = 0.0
    recycle_risk_fraction: float = 0.0
    chemical_fraction: float = 0.0
    product_fraction: float = 1.0


@dataclass
class ThermalSlice:
    volume_l: float
    temp_c: float
    residence_time_s: float
    relative_lethality: float
    thermal_safe: bool
    pressure_safe: bool = True
    contamination_risk: bool = False
    process_differential_pressure_bar: float = 0.0
    fastest_residence_time_s: float = 0.0
    product_temp_c: float | None = None
    downstream_fault_active: bool = False
    downstream_regenerator_factor: float = 1.0
    downstream_cooler_factor: float = 1.0
    exponential_was_bounded: bool = False
    mean_pass_count: float = 0.0
    recycle_risk_fraction: float = 0.0
    chemical_fraction: float = 0.0
    product_fraction: float = 1.0


@dataclass(frozen=True)
class HoldingOutlet:
    slices: tuple[ThermalSlice, ...]
    volume_l: float
    mean_temp_c: float
    mean_residence_time_s: float
    mean_relative_lethality: float
    thermal_safe_fraction: float
    mean_fastest_residence_time_s: float = 0.0
    bounded_exponential_count: int = 0


@dataclass(frozen=True)
class ScenarioModifiers:
    flow_factor: float = 1.0
    heater_capacity_factor: float = 1.0
    regenerator_factor: float = 1.0
    cooler_factor: float = 1.0
    booster_factor: float = 1.0
    control_sensor_bias_c: float = 0.0
    safety_sensor_bias_c: float = 0.0
    flow_meter_bias_fraction: float = 0.0
    pressure_sensor_bias_bar: float = 0.0
    leak_fraction: float = 0.0
    fouling_rate_factor: float = 1.0
    valve_stuck_forward: bool = False
    power_available: bool = True
    sensor_available: bool = True
    valve_travel_time_factor: float = 1.0
    valve_leakage_fraction: float = 0.0
    valve_feedback_bias_fraction: float = 0.0
    cleaning_effectiveness_factor: float = 1.0


SCENARIO_NAMES = (
    "normal",
    "steam_loss",
    "flow_surge",
    "sensor_bias_high",
    "control_sensor_bias_high",
    "safety_sensor_bias_high",
    "dual_sensor_common_bias",
    "flowmeter_bias_low_with_surge",
    "booster_pump_failure",
    "regenerator_leak_pressure_inversion",
    "progressive_fouling",
    "valve_stuck_forward_steam_loss",
    "cooling_utility_loss",
    "power_failure",
    "cip_cycle",
    "start_stop",
    "incomplete_cleaning",
    "sensor_drift",
    "sensor_dropout",
    "slow_valve",
    "valve_leakage",
)


ALARM_COLUMNS = (
    "alarm_low_temperature",
    "alarm_low_holding_time",
    "alarm_high_flow",
    "alarm_low_pressure_differential",
    "alarm_sensor_disagreement",
    "alarm_regenerator_leak",
    "alarm_fdv_mismatch",
    "alarm_high_product_temperature",
    "alarm_sensor_dropout",
)

DIAGNOSTIC_ALARM_COLUMNS = ("alarm_unsafe_forward", "alarm_high_fouling")
ALL_ALARM_COLUMNS = ALARM_COLUMNS + DIAGNOSTIC_ALARM_COLUMNS


def _config_hash(config: HTSTConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fault_active(config: HTSTConfig, time_s: float) -> bool:
    return config.fault_start_s <= time_s < config.fault_start_s + config.fault_duration_s


def scenario_modifiers(
    name: str,
    time_s: float,
    config: HTSTConfig | None = None,
) -> ScenarioModifiers:
    cfg = config or HTSTConfig()
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {name!r}; choose from {SCENARIO_NAMES}")
    active = _fault_active(cfg, time_s)
    severity = cfg.fault_severity
    if active and severity <= 0.0:
        return ScenarioModifiers()
    if name == "steam_loss" and active:
        return ScenarioModifiers(heater_capacity_factor=max(0.0, 1.0 - severity))
    if name == "flow_surge" and active:
        return ScenarioModifiers(flow_factor=1.0 + 0.50 * severity)
    if name in {"sensor_bias_high", "control_sensor_bias_high"} and active:
        return ScenarioModifiers(control_sensor_bias_c=3.0 * severity)
    if name == "safety_sensor_bias_high" and active:
        return ScenarioModifiers(safety_sensor_bias_c=3.0 * severity)
    if name == "dual_sensor_common_bias" and active:
        return ScenarioModifiers(
            control_sensor_bias_c=3.0 * severity,
            safety_sensor_bias_c=3.0 * severity,
        )
    if name == "flowmeter_bias_low_with_surge" and active:
        return ScenarioModifiers(
            flow_factor=1.0 + 0.35 * severity,
            flow_meter_bias_fraction=-0.30 * severity,
        )
    if name == "booster_pump_failure" and active:
        return ScenarioModifiers(booster_factor=max(0.0, 1.0 - severity))
    if name == "regenerator_leak_pressure_inversion" and active:
        return ScenarioModifiers(
            booster_factor=max(0.0, 1.0 - 0.90 * severity),
            leak_fraction=0.02 * severity,
        )
    if name == "progressive_fouling" and active:
        return ScenarioModifiers(fouling_rate_factor=1.0 + 79.0 * severity)
    if name == "valve_stuck_forward_steam_loss" and active:
        return ScenarioModifiers(
            heater_capacity_factor=max(0.0, 1.0 - severity),
            valve_stuck_forward=True,
        )
    if name == "cooling_utility_loss" and active:
        return ScenarioModifiers(cooler_factor=max(0.0, 1.0 - 0.90 * severity))
    if name == "power_failure" and active:
        return ScenarioModifiers(
            flow_factor=0.0,
            heater_capacity_factor=0.0,
            booster_factor=0.0,
            power_available=False,
        )
    if name == "start_stop" and active:
        return ScenarioModifiers(
            flow_factor=0.0,
            heater_capacity_factor=0.0,
            booster_factor=0.0,
        )
    if name == "incomplete_cleaning" and active:
        return ScenarioModifiers(
            cleaning_effectiveness_factor=max(0.0, 1.0 - 0.80 * severity),
        )
    if name == "sensor_drift" and active:
        duration = max(cfg.fault_duration_s, cfg.dt_s)
        progress = min(1.0, max(0.0, (time_s - cfg.fault_start_s) / duration))
        return ScenarioModifiers(control_sensor_bias_c=3.0 * severity * progress)
    if name == "sensor_dropout" and active:
        return ScenarioModifiers(sensor_available=False)
    if name == "slow_valve" and active:
        return ScenarioModifiers(
            heater_capacity_factor=max(0.0, 1.0 - severity),
            valve_travel_time_factor=1.0 + 7.0 * severity,
        )
    if name == "valve_leakage" and active:
        return ScenarioModifiers(
            heater_capacity_factor=max(0.0, 1.0 - severity),
            valve_leakage_fraction=min(0.25, 0.08 * severity),
        )
    return ScenarioModifiers()


def _bounded_exp(argument: float, limit: float) -> tuple[float, bool]:
    """Return a finite exponential and whether its argument was clipped."""
    if not math.isfinite(argument):
        raise FloatingPointError(f"non-finite exponential argument: {argument}")
    bounded = min(limit, max(-limit, argument))
    return math.exp(bounded), bounded != argument


def _relative_treatment_diagnostic(
    residence_time_s: float,
    temp_c: float,
    config: HTSTConfig,
) -> tuple[float, bool]:
    """Generic dimensionless time-temperature diagnostic with bounded exponent."""
    if residence_time_s <= 0.0:
        return 0.0, False
    log_value = math.log(residence_time_s / config.lethality_reference_time_s)
    log_value += (
        (temp_c - config.lethality_reference_temp_c)
        / config.lethality_z_c
        * math.log(10.0)
    )
    return _bounded_exp(log_value, config.max_exponential_argument)


def _ensure_finite_mapping(values: dict[str, object], context: str) -> None:
    for name, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"{context}.{name} is not finite: {value}")


def cip_phase(time_s: float) -> tuple[str, float, float, float, float]:
    """Return phase, inlet temp, setpoint, flow factor and concentration %."""
    if time_s < 60.0:
        return "CIP_PRE_RINSE", 20.0, 25.0, 0.80, 0.0
    if time_s < 240.0:
        return "CIP_CAUSTIC", 25.0, 75.0, 0.85, 2.0
    if time_s < 300.0:
        return "CIP_INTERMEDIATE_RINSE", 20.0, 30.0, 0.80, 0.0
    if time_s < 420.0:
        return "CIP_ACID", 25.0, 65.0, 0.85, 1.0
    if time_s < 540.0:
        return "CIP_FINAL_RINSE", 20.0, 25.0, 0.80, 0.0
    return "CIP_COMPLETE", 20.0, 20.0, 0.0, 0.0


def _weighted(slices: Iterable[ThermalSlice]) -> tuple[float, float, float, float, float]:
    items = list(slices)
    volume = sum(item.volume_l for item in items)
    if volume <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    temp = sum(item.volume_l * item.temp_c for item in items) / volume
    residence = sum(item.volume_l * item.residence_time_s for item in items) / volume
    lethality = sum(item.volume_l * item.relative_lethality for item in items) / volume
    safe_fraction = sum(item.volume_l for item in items if item.thermal_safe) / volume
    return volume, temp, residence, lethality, safe_fraction


def _weighted_fastest_residence(slices: Iterable[ThermalSlice]) -> float:
    items = list(slices)
    volume = sum(item.volume_l for item in items)
    if volume <= 0.0:
        return 0.0
    return sum(item.volume_l * item.fastest_residence_time_s for item in items) / volume


def _weighted_product_temp(slices: Iterable[ThermalSlice], fallback_c: float) -> float:
    items = list(slices)
    volume = sum(item.volume_l for item in items)
    if volume <= 0.0:
        return fallback_c
    return (
        sum(
            item.volume_l
            * (item.product_temp_c if item.product_temp_c is not None else item.temp_c)
            for item in items
        )
        / volume
    )


def _weighted_attribute(
    slices: Iterable[ThermalSlice],
    attribute: str,
    fallback: float = 0.0,
) -> float:
    items = list(slices)
    volume = sum(item.volume_l for item in items)
    if volume <= 0.0:
        return fallback
    return sum(
        item.volume_l * float(getattr(item, attribute)) for item in items
    ) / volume


def _slice_window(
    slices: Iterable[ThermalSlice],
    start_fraction: float,
    end_fraction: float,
) -> tuple[ThermalSlice, ...]:
    """Return the chronological volume window [start, end] of a slice stream."""
    items = list(slices)
    total_volume_l = sum(item.volume_l for item in items)
    start = min(1.0, max(0.0, start_fraction)) * total_volume_l
    end = min(1.0, max(0.0, end_fraction)) * total_volume_l
    if total_volume_l <= 0.0 or end <= start + 1e-12:
        return ()
    selected: list[ThermalSlice] = []
    cursor = 0.0
    for item in items:
        item_end = cursor + item.volume_l
        overlap = max(0.0, min(item_end, end) - max(cursor, start))
        if overlap > 1e-12:
            selected.append(replace(item, volume_l=overlap))
        cursor = item_end
        if cursor >= end - 1e-12:
            break
    return tuple(selected)


class HTSTSimulator:
    def __init__(self, config: HTSTConfig | None = None) -> None:
        self.config = config or HTSTConfig()
        self.reset()

    def _initial_holding_queue(self) -> Deque[FluidParcel]:
        cfg = self.config
        queue: Deque[FluidParcel] = collections.deque()
        remaining_time = cfg.nominal_holding_time_s
        cursor = -cfg.nominal_holding_time_s
        while remaining_time > 1e-12:
            duration = min(cfg.dt_s, remaining_time)
            queue.append(
                FluidParcel(
                    volume_l=cfg.nominal_flow_l_s * duration,
                    temp_c=cfg.raw_milk_temp_c,
                    entry_start_s=cursor,
                    entry_end_s=cursor + duration,
                )
            )
            cursor += duration
            remaining_time -= duration
        return queue

    def _initial_fdv_queue(self) -> Deque[ThermalSlice]:
        cfg = self.config
        queue: Deque[ThermalSlice] = collections.deque()
        remaining = cfg.fdv_line_volume_l
        chunk = cfg.nominal_flow_l_s * cfg.dt_s
        while remaining > 1e-12:
            take = min(chunk, remaining)
            queue.append(ThermalSlice(take, cfg.raw_milk_temp_c, 0.0, 0.0, False))
            remaining -= take
        return queue

    def _initial_post_fdv_queue(self) -> Deque[ThermalSlice]:
        cfg = self.config
        queue: Deque[ThermalSlice] = collections.deque()
        remaining = cfg.post_fdv_inventory_l
        chunk = max(cfg.nominal_flow_l_s * cfg.dt_s, _VOLUME_EPSILON_L)
        fastest_residence = cfg.nominal_holding_time_s * cfg.fastest_flow_efficiency
        while remaining > 1e-12:
            take = min(chunk, remaining)
            queue.append(
                ThermalSlice(
                    volume_l=take,
                    temp_c=cfg.pasteurization_setpoint_c,
                    residence_time_s=cfg.nominal_holding_time_s,
                    relative_lethality=1.0,
                    thermal_safe=True,
                    pressure_safe=True,
                    contamination_risk=False,
                    process_differential_pressure_bar=cfg.required_pressure_differential_bar,
                    fastest_residence_time_s=fastest_residence,
                    product_temp_c=cfg.final_product_target_c,
                )
            )
            remaining -= take
        return queue

    @property
    def fdv_forward(self) -> bool:
        """Compatibility boolean: true while any forward opening remains."""
        return self._fdv_position > 1.0e-12

    @fdv_forward.setter
    def fdv_forward(self, value: bool) -> None:
        self._fdv_position = 1.0 if bool(value) else 0.0
        if hasattr(self, "forward_permissive_confirmed"):
            self.forward_permissive_confirmed = bool(value)

    def reset(self) -> None:
        cfg = self.config
        self.rng = random.Random(cfg.random_seed)
        self._campaign_time_s = 0.0
        self._campaign_active = False
        self._last_phase_kind: str | None = None
        self._restart_guard_active = False
        self._cip_release_permissive = True
        self._last_cip_conductivity_proxy_ms_cm = 0.2
        self._last_cip_ph_proxy = 7.0
        self._surface_hygiene_risk_fraction = 0.0
        self._surface_state_synchronized = False
        self.preheat_temp_c = cfg.raw_milk_temp_c
        self.heater_temp_c = cfg.raw_milk_temp_c
        self.last_hold_temp_c = cfg.raw_milk_temp_c
        self.last_control_sensor_c = cfg.raw_milk_temp_c
        self.controller_integral = 0.0
        self.steam_valve = 0.0
        self._fdv_position = 0.0
        self.fdv_position_feedback = 0.0
        self.fdv_mismatch_time_s = 0.0
        self.forward_permissive_confirmed = False
        self.pending_forward_safe_volume_l = 0.0
        self.pending_divert_time_s = 0.0
        self.ever_forward = False
        self.last_routed_forward = False
        self.fouling_index = cfg.initial_fouling_index
        self.holding_queue = self._initial_holding_queue()
        self.fdv_line_queue = self._initial_fdv_queue()
        self.post_fdv_queue = self._initial_post_fdv_queue()
        self.last_product_temp_c = cfg.final_product_target_c
        self.last_preheat_sensor_c = cfg.raw_milk_temp_c
        self.last_safety_sensor_c = cfg.raw_milk_temp_c
        self.balance_tank = BalanceTank(
            BalanceTankConfig(
                capacity_l=cfg.balance_tank_capacity_l,
                initial_volume_l=cfg.balance_tank_initial_volume_l,
                minimum_operating_volume_l=cfg.balance_tank_minimum_volume_l,
                initial_temperature_c=cfg.raw_milk_temp_c,
                ambient_temperature_c=cfg.ambient_temp_c,
                heat_loss_tau_s=cfg.balance_tank_heat_loss_tau_s,
                density_kg_l=cfg.density_kg_l,
                heat_capacity_kj_kg_k=cfg.heat_capacity_kj_kg_k,
            )
        )
        self.pending_return_stream = LiquidStream(0.0, cfg.raw_milk_temp_c)
        self.cip_soil_model = CIPSoilModel(
            CIPSoilConfig(
                initial_protein_soil_g=cfg.cip_initial_protein_soil_g,
                initial_mineral_soil_g=cfg.cip_initial_mineral_soil_g,
                protein_reference_rate_s=cfg.cip_caustic_removal_rate_s,
                mineral_reference_rate_s=cfg.cip_acid_removal_rate_s,
                reference_velocity_m_s=cfg.cip_reference_velocity_m_s,
                line_hold_up_l=cfg.cip_line_hold_up_l,
                clean_soil_threshold_g=cfg.cip_clean_soil_threshold_g,
                residual_chemical_threshold_fraction=(
                    cfg.cip_residual_chemical_threshold_fraction
                ),
                ambient_temperature_c=cfg.ambient_temp_c,
            )
        )
        # Production and campaign runs begin from the configured fouling state.
        # Standalone CIP runs explicitly seed their historical 650/350 g load
        # in ``run_phase`` so the legacy numerical path remains unchanged.
        self._set_surface_from_fouling(cfg.initial_fouling_index)
        hx_common = {
            "product_holdup_l": cfg.hx_product_holdup_l,
            "utility_holdup_l": cfg.hx_utility_holdup_l,
            "wall_heat_capacity_kj_k": cfg.hx_wall_heat_capacity_kj_k,
            "product_density_kg_l": cfg.density_kg_l,
            "product_heat_capacity_kj_kg_k": cfg.heat_capacity_kj_kg_k,
            "ambient_ua_kw_k": cfg.hx_ambient_ua_kw_k,
            "ambient_temperature_c": cfg.ambient_temp_c,
            "fouling_resistance_ratio_at_one": (
                cfg.hx_fouling_resistance_ratio_at_one
            ),
            "internal_max_dt_s": cfg.hx_internal_max_dt_s,
        }
        self.regenerator_hx = DynamicHeatExchanger(
            HeatExchangerConfig(
                clean_ua_kw_k=cfg.hx_regenerator_clean_ua_kw_k,
                utility_density_kg_l=cfg.density_kg_l,
                utility_heat_capacity_kj_kg_k=cfg.heat_capacity_kj_kg_k,
                **hx_common,
            ),
            initial_product_c=cfg.raw_milk_temp_c,
            initial_utility_c=cfg.pasteurization_setpoint_c,
        )
        self.heater_hx = DynamicHeatExchanger(
            HeatExchangerConfig(
                clean_ua_kw_k=cfg.hx_heater_clean_ua_kw_k,
                **hx_common,
            ),
            initial_product_c=cfg.raw_milk_temp_c,
            initial_utility_c=cfg.heating_utility_temp_c,
        )
        self.cooler_hx = DynamicHeatExchanger(
            HeatExchangerConfig(
                clean_ua_kw_k=cfg.hx_cooler_clean_ua_kw_k,
                **hx_common,
            ),
            initial_product_c=cfg.pasteurization_setpoint_c,
            initial_utility_c=cfg.cooling_utility_temp_c,
        )

    @property
    def _surface_soil_scale_g(self) -> float:
        cfg = self.config
        if cfg.cip_initial_fouling_index <= 0.0:
            raise ValueError(
                "cip_initial_fouling_index must be positive for campaign soil mapping"
            )
        return (
            cfg.cip_initial_protein_soil_g + cfg.cip_initial_mineral_soil_g
        ) / cfg.cip_initial_fouling_index

    def _set_surface_from_fouling(self, fouling_index: float) -> None:
        """Synchronize the two-soil state to the canonical fouling scalar."""
        index = min(1.0, max(0.0, float(fouling_index)))
        total = index * self._surface_soil_scale_g
        cfg = self.config
        configured_total = (
            cfg.cip_initial_protein_soil_g + cfg.cip_initial_mineral_soil_g
        )
        protein_share = cfg.cip_initial_protein_soil_g / configured_total
        self.cip_soil_model.set_soil(
            total * protein_share,
            total * (1.0 - protein_share),
        )
        self.fouling_index = index
        self._surface_state_synchronized = True

    def _set_fouling_from_surface(self) -> None:
        cfg = self.config
        initial_soil_g = (
            cfg.cip_initial_protein_soil_g + cfg.cip_initial_mineral_soil_g
        )
        self.fouling_index = min(
            1.0,
            cfg.cip_initial_fouling_index
            * self.cip_soil_model.total_soil_g
            / initial_soil_g,
        )
        self._surface_state_synchronized = True

    @staticmethod
    def _state_digest(payload: Mapping[str, object]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def export_state(self) -> dict[str, object]:
        """Export every dynamic state needed for an exact JSON round trip."""
        rng_version, rng_internal, gauss_next = self.rng.getstate()
        payload: dict[str, object] = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "config_hash": _config_hash(self.config),
            "campaign_time_s": self._campaign_time_s,
            "campaign_active": self._campaign_active,
            "last_phase_kind": self._last_phase_kind,
            "restart_guard_active": self._restart_guard_active,
            "cip_release_permissive": self._cip_release_permissive,
            "surface_state_synchronized": self._surface_state_synchronized,
            "rng_state": {
                "version": rng_version,
                "internal": list(rng_internal),
                "gauss_next": gauss_next,
            },
            "scalars": {
                "preheat_temp_c": self.preheat_temp_c,
                "heater_temp_c": self.heater_temp_c,
                "last_hold_temp_c": self.last_hold_temp_c,
                "last_control_sensor_c": self.last_control_sensor_c,
                "controller_integral": self.controller_integral,
                "steam_valve": self.steam_valve,
                "fdv_position": self._fdv_position,
                "fdv_position_feedback": self.fdv_position_feedback,
                "fdv_mismatch_time_s": self.fdv_mismatch_time_s,
                "pending_forward_safe_volume_l": self.pending_forward_safe_volume_l,
                "pending_divert_time_s": self.pending_divert_time_s,
                "fouling_index": self.fouling_index,
                "last_product_temp_c": self.last_product_temp_c,
                "last_preheat_sensor_c": self.last_preheat_sensor_c,
                "last_safety_sensor_c": self.last_safety_sensor_c,
                "last_cip_conductivity_proxy_ms_cm": (
                    self._last_cip_conductivity_proxy_ms_cm
                ),
                "last_cip_ph_proxy": self._last_cip_ph_proxy,
                "surface_hygiene_risk_fraction": (
                    self._surface_hygiene_risk_fraction
                ),
            },
            "booleans": {
                "forward_permissive_confirmed": self.forward_permissive_confirmed,
                "ever_forward": self.ever_forward,
                "last_routed_forward": self.last_routed_forward,
            },
            "holding_queue": [asdict(item) for item in self.holding_queue],
            "fdv_line_queue": [asdict(item) for item in self.fdv_line_queue],
            "post_fdv_queue": [asdict(item) for item in self.post_fdv_queue],
            "balance_tank": self.balance_tank.export_state(),
            "pending_return_stream": asdict(self.pending_return_stream),
            "cip_soil": self.cip_soil_model.export_state(),
            "heat_exchangers": {
                "regenerator": self.regenerator_hx.export_state(),
                "heater": self.heater_hx.export_state(),
                "cooler": self.cooler_hx.export_state(),
            },
        }
        return {**payload, "state_sha256": self._state_digest(payload)}

    @staticmethod
    def _finite_state_number(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"state {name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"state {name} must be finite")
        return number

    def _apply_validated_state(self, payload: Mapping[str, object]) -> None:
        expected = {
            "state_schema_version",
            "model_version",
            "config_hash",
            "campaign_time_s",
            "campaign_active",
            "last_phase_kind",
            "restart_guard_active",
            "cip_release_permissive",
            "surface_state_synchronized",
            "rng_state",
            "scalars",
            "booleans",
            "holding_queue",
            "fdv_line_queue",
            "post_fdv_queue",
            "balance_tank",
            "pending_return_stream",
            "cip_soil",
            "heat_exchangers",
        }
        if set(payload) != expected:
            raise ValueError("invalid simulator state fields")
        if payload["state_schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported simulator state schema version")
        if payload["model_version"] != MODEL_VERSION:
            raise ValueError("simulator state model version mismatch")
        if payload["config_hash"] != _config_hash(self.config):
            raise ValueError("simulator state config hash mismatch")

        campaign_time = self._finite_state_number(
            "campaign_time_s", payload["campaign_time_s"]
        )
        if campaign_time < 0.0:
            raise ValueError("state campaign_time_s must be non-negative")
        for name in (
            "campaign_active",
            "restart_guard_active",
            "cip_release_permissive",
            "surface_state_synchronized",
        ):
            if not isinstance(payload[name], bool):
                raise ValueError(f"state {name} must be boolean")
        last_phase_kind = payload["last_phase_kind"]
        if last_phase_kind not in {None, "production", "cip"}:
            raise ValueError("invalid last_phase_kind in simulator state")

        rng_state = payload["rng_state"]
        if not isinstance(rng_state, Mapping) or set(rng_state) != {
            "version",
            "internal",
            "gauss_next",
        }:
            raise ValueError("invalid RNG state structure")
        version = rng_state["version"]
        internal = rng_state["internal"]
        gauss_next = rng_state["gauss_next"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("invalid RNG state version")
        if not isinstance(internal, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in internal
        ):
            raise ValueError("invalid RNG internal state")
        if gauss_next is not None:
            gauss_next = self._finite_state_number("rng.gauss_next", gauss_next)
        restored_rng = random.Random()
        try:
            restored_rng.setstate((version, tuple(internal), gauss_next))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid RNG state") from exc

        scalar_names = {
            "preheat_temp_c",
            "heater_temp_c",
            "last_hold_temp_c",
            "last_control_sensor_c",
            "controller_integral",
            "steam_valve",
            "fdv_position",
            "fdv_position_feedback",
            "fdv_mismatch_time_s",
            "pending_forward_safe_volume_l",
            "pending_divert_time_s",
            "fouling_index",
            "last_product_temp_c",
            "last_preheat_sensor_c",
            "last_safety_sensor_c",
            "last_cip_conductivity_proxy_ms_cm",
            "last_cip_ph_proxy",
            "surface_hygiene_risk_fraction",
        }
        scalars = payload["scalars"]
        if not isinstance(scalars, Mapping) or set(scalars) != scalar_names:
            raise ValueError("invalid simulator scalar state")
        numbers = {
            name: self._finite_state_number(name, scalars[name])
            for name in scalar_names
        }
        for name in (
            "steam_valve",
            "fdv_position",
            "fdv_position_feedback",
            "fouling_index",
            "surface_hygiene_risk_fraction",
        ):
            if not 0.0 <= numbers[name] <= 1.0:
                raise ValueError(f"state {name} must be in [0, 1]")
        for name in (
            "fdv_mismatch_time_s",
            "pending_forward_safe_volume_l",
            "pending_divert_time_s",
        ):
            if numbers[name] < 0.0:
                raise ValueError(f"state {name} must be non-negative")

        booleans = payload["booleans"]
        boolean_names = {
            "forward_permissive_confirmed",
            "ever_forward",
            "last_routed_forward",
        }
        if (
            not isinstance(booleans, Mapping)
            or set(booleans) != boolean_names
            or not all(isinstance(booleans[name], bool) for name in boolean_names)
        ):
            raise ValueError("invalid simulator boolean state")

        def restore_queue(
            raw: object,
            cls: type[FluidParcel] | type[ThermalSlice],
            label: str,
        ) -> Deque[FluidParcel] | Deque[ThermalSlice]:
            if not isinstance(raw, list):
                raise ValueError(f"state {label} must be a list")
            expected_fields = set(cls.__dataclass_fields__)
            restored: Deque[FluidParcel] | Deque[ThermalSlice] = collections.deque()
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping) or set(item) != expected_fields:
                    raise ValueError(f"invalid {label}[{index}] fields")
                try:
                    value = cls(**dict(item))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {label}[{index}]") from exc
                for field_name, field_value in asdict(value).items():
                    if field_value is None:
                        continue
                    if isinstance(field_value, bool):
                        continue
                    number = self._finite_state_number(
                        f"{label}[{index}].{field_name}", field_value
                    )
                    if field_name == "volume_l" and number <= 0.0:
                        raise ValueError(f"{label}[{index}] volume must be positive")
                    if field_name.endswith("fraction") and not 0.0 <= number <= 1.0:
                        raise ValueError(
                            f"{label}[{index}].{field_name} must be in [0, 1]"
                        )
                if isinstance(value, FluidParcel):
                    if value.entry_end_s < value.entry_start_s:
                        raise ValueError(f"{label}[{index}] has reversed timestamps")
                    if value.entry_end_s > campaign_time + 1e-9:
                        raise ValueError(f"{label}[{index}] begins in the future")
                    if value.mean_pass_count < 0.0:
                        raise ValueError(f"{label}[{index}] pass count is negative")
                else:
                    for field_name in (
                        "residence_time_s",
                        "relative_lethality",
                        "fastest_residence_time_s",
                        "mean_pass_count",
                    ):
                        if getattr(value, field_name) < 0.0:
                            raise ValueError(
                                f"{label}[{index}].{field_name} is negative"
                            )
                    for field_name in (
                        "downstream_regenerator_factor",
                        "downstream_cooler_factor",
                    ):
                        if not 0.0 <= getattr(value, field_name) <= 1.0:
                            raise ValueError(
                                f"{label}[{index}].{field_name} must be in [0, 1]"
                            )
                restored.append(value)
            if not restored:
                raise ValueError(f"state {label} cannot be empty")
            return restored

        holding = restore_queue(payload["holding_queue"], FluidParcel, "holding_queue")
        fdv = restore_queue(payload["fdv_line_queue"], ThermalSlice, "fdv_line_queue")
        post = restore_queue(payload["post_fdv_queue"], ThermalSlice, "post_fdv_queue")
        inventories = (
            (holding, self.config.holding_tube_volume_l, "holding_queue"),
            (fdv, self.config.fdv_line_volume_l, "fdv_line_queue"),
            (post, self.config.post_fdv_inventory_l, "post_fdv_queue"),
        )
        for queue, expected_volume, label in inventories:
            actual = sum(item.volume_l for item in queue)
            if abs(actual - expected_volume) > 1e-7 * max(1.0, expected_volume):
                raise ValueError(f"state {label} inventory mismatch")

        pending_raw = payload["pending_return_stream"]
        if not isinstance(pending_raw, Mapping):
            raise ValueError("invalid pending return stream")
        try:
            pending = LiquidStream(**dict(pending_raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid pending return stream") from exc

        balance = payload["balance_tank"]
        soil = payload["cip_soil"]
        exchangers = payload["heat_exchangers"]
        if not isinstance(balance, dict) or not isinstance(soil, dict):
            raise ValueError("invalid component state structure")
        if not isinstance(exchangers, Mapping) or set(exchangers) != {
            "regenerator",
            "heater",
            "cooler",
        }:
            raise ValueError("invalid heat-exchanger state structure")
        self.balance_tank.import_state(balance)
        self.cip_soil_model.import_state(soil)
        for name, component in (
            ("regenerator", self.regenerator_hx),
            ("heater", self.heater_hx),
            ("cooler", self.cooler_hx),
        ):
            raw = exchangers[name]
            if not isinstance(raw, dict):
                raise ValueError(f"invalid {name} heat-exchanger state")
            component.import_state(raw)
        cfg = self.config
        initial_soil_g = (
            cfg.cip_initial_protein_soil_g + cfg.cip_initial_mineral_soil_g
        )
        expected_fouling = min(
            1.0,
            cfg.cip_initial_fouling_index
            * self.cip_soil_model.total_soil_g
            / initial_soil_g,
        )
        if abs(numbers["fouling_index"] - expected_fouling) > 1e-10:
            raise ValueError("surface soil and fouling_index are inconsistent")

        self.rng = restored_rng
        self._campaign_time_s = campaign_time
        self._campaign_active = bool(payload["campaign_active"])
        self._last_phase_kind = last_phase_kind
        self._restart_guard_active = bool(payload["restart_guard_active"])
        self._cip_release_permissive = bool(payload["cip_release_permissive"])
        self._surface_state_synchronized = bool(
            payload["surface_state_synchronized"]
        )
        self.preheat_temp_c = numbers["preheat_temp_c"]
        self.heater_temp_c = numbers["heater_temp_c"]
        self.last_hold_temp_c = numbers["last_hold_temp_c"]
        self.last_control_sensor_c = numbers["last_control_sensor_c"]
        self.controller_integral = numbers["controller_integral"]
        self.steam_valve = numbers["steam_valve"]
        self._fdv_position = numbers["fdv_position"]
        self.fdv_position_feedback = numbers["fdv_position_feedback"]
        self.fdv_mismatch_time_s = numbers["fdv_mismatch_time_s"]
        self.pending_forward_safe_volume_l = numbers[
            "pending_forward_safe_volume_l"
        ]
        self.pending_divert_time_s = numbers["pending_divert_time_s"]
        self.fouling_index = numbers["fouling_index"]
        self.last_product_temp_c = numbers["last_product_temp_c"]
        self.last_preheat_sensor_c = numbers["last_preheat_sensor_c"]
        self.last_safety_sensor_c = numbers["last_safety_sensor_c"]
        self._last_cip_conductivity_proxy_ms_cm = numbers[
            "last_cip_conductivity_proxy_ms_cm"
        ]
        self._last_cip_ph_proxy = numbers["last_cip_ph_proxy"]
        self._surface_hygiene_risk_fraction = numbers[
            "surface_hygiene_risk_fraction"
        ]
        self.forward_permissive_confirmed = bool(
            booleans["forward_permissive_confirmed"]
        )
        self.ever_forward = bool(booleans["ever_forward"])
        self.last_routed_forward = bool(booleans["last_routed_forward"])
        self.holding_queue = holding  # type: ignore[assignment]
        self.fdv_line_queue = fdv  # type: ignore[assignment]
        self.post_fdv_queue = post  # type: ignore[assignment]
        self.pending_return_stream = pending

    def import_state(self, state: Mapping[str, object]) -> None:
        """Atomically validate and restore a versioned campaign checkpoint."""
        if not isinstance(state, Mapping):
            raise ValueError("simulator state must be a mapping")
        supplied = dict(state)
        digest = supplied.pop("state_sha256", None)
        if not isinstance(digest, str) or digest != self._state_digest(supplied):
            raise ValueError("simulator state checksum mismatch")
        candidate = HTSTSimulator(self.config)
        candidate._apply_validated_state(supplied)
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _thermal_relaxation_alpha(self, step_dt_s: float) -> float:
        value, _ = _bounded_exp(
            -step_dt_s / self.config.stationary_cooling_tau_s,
            self.config.max_exponential_argument,
        )
        return 1.0 - value

    def _relax_stationary_holding(self, step_dt_s: float) -> None:
        cfg = self.config
        alpha = self._thermal_relaxation_alpha(step_dt_s)
        for parcel in self.holding_queue:
            parcel.temp_c += (cfg.ambient_temp_c - parcel.temp_c) * alpha
        self.last_hold_temp_c += (
            cfg.ambient_temp_c - self.last_hold_temp_c
        ) * alpha

    def _relax_stationary_downstream(self, step_dt_s: float) -> None:
        cfg = self.config
        alpha = self._thermal_relaxation_alpha(step_dt_s)
        for queue in (self.fdv_line_queue, self.post_fdv_queue):
            self._relax_slice_queue(queue, alpha)

    def _relax_slice_queue(
        self,
        queue: Deque[ThermalSlice],
        alpha: float,
    ) -> None:
        cfg = self.config
        for item in queue:
            item.temp_c += (cfg.ambient_temp_c - item.temp_c) * alpha
            if item.product_temp_c is not None:
                item.product_temp_c += (
                    cfg.ambient_temp_c - item.product_temp_c
                ) * alpha

    def _pi_control(
        self,
        measured_temp_c: float,
        measured_preheat_c: float,
        setpoint_c: float,
        step_dt_s: float,
    ) -> float:
        cfg = self.config
        for name, value in (
            ("measured_temp_c", measured_temp_c),
            ("measured_preheat_c", measured_preheat_c),
            ("setpoint_c", setpoint_c),
            ("step_dt_s", step_dt_s),
        ):
            if not math.isfinite(value):
                raise FloatingPointError(f"{name} is not finite")
        error = setpoint_c - measured_temp_c
        feed_forward = max(0.0, setpoint_c - measured_preheat_c) / cfg.heater_max_delta_c
        proposed_integral = self.controller_integral + error * step_dt_s
        proposed_integral = min(1.0e9, max(-1.0e9, proposed_integral))
        raw = feed_forward + cfg.controller_kp * error + cfg.controller_ki * proposed_integral
        if not math.isfinite(raw):
            raise FloatingPointError("PI controller output is not finite")
        output = min(1.0, max(0.0, raw))
        if output == raw or (output >= 1.0 and error < 0.0) or (output <= 0.0 and error > 0.0):
            self.controller_integral = proposed_integral
        return output

    def _move_through_holding_tube(
        self,
        inlet_volume_l: float,
        inlet_temp_c: float,
        time_start_s: float,
        time_end_s: float,
        *,
        mean_pass_count: float = 0.0,
        recycle_risk_fraction: float = 0.0,
        chemical_fraction: float = 0.0,
        product_fraction: float = 1.0,
    ) -> HoldingOutlet:
        if inlet_volume_l < 0.0 or not math.isfinite(inlet_volume_l):
            raise ValueError("inlet volume must be finite and nonnegative")
        if not math.isfinite(inlet_temp_c):
            raise ValueError("inlet temperature must be finite")
        if not math.isfinite(time_start_s) or not math.isfinite(time_end_s):
            raise ValueError("holding interval endpoints must be finite")
        for name, value in (
            ("mean_pass_count", mean_pass_count),
            ("recycle_risk_fraction", recycle_risk_fraction),
            ("chemical_fraction", chemical_fraction),
            ("product_fraction", product_fraction),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            recycle_risk_fraction > 1.0
            or chemical_fraction > 1.0
            or product_fraction > 1.0
        ):
            raise ValueError("holding stream fractions must be <= 1")
        if time_end_s < time_start_s:
            raise ValueError("holding interval end must not precede its start")
        if inlet_volume_l == 0.0:
            self._relax_stationary_holding(time_end_s - time_start_s)
            return HoldingOutlet(
                (), 0.0, self.last_hold_temp_c, 0.0, 0.0, 0.0, 0.0, 0
            )
        self.holding_queue.append(
            FluidParcel(
                inlet_volume_l,
                inlet_temp_c,
                time_start_s,
                time_end_s,
                mean_pass_count,
                recycle_risk_fraction,
                chemical_fraction,
                product_fraction,
            )
        )
        remaining = inlet_volume_l
        outlet_consumed = 0.0
        interval = time_end_s - time_start_s
        slices: list[ThermalSlice] = []
        cfg = self.config
        while remaining > 1e-10:
            parcel = self.holding_queue[0]
            available_before = parcel.volume_l
            take = min(remaining, available_before)
            fraction = take / available_before
            entry_cut = parcel.entry_start_s + (
                parcel.entry_end_s - parcel.entry_start_s
            ) * fraction
            entry_mid = 0.5 * (parcel.entry_start_s + entry_cut)
            exit_start = time_start_s + interval * outlet_consumed / inlet_volume_l
            exit_end = time_start_s + interval * (outlet_consumed + take) / inlet_volume_l
            exit_mid = 0.5 * (exit_start + exit_end)
            residence = max(0.0, exit_mid - entry_mid)
            fastest_residence = residence * cfg.fastest_flow_efficiency
            lethality, exponential_was_bounded = _relative_treatment_diagnostic(
                fastest_residence,
                parcel.temp_c,
                cfg,
            )
            thermal_safe = (
                parcel.temp_c >= cfg.diversion_threshold_c
                and fastest_residence >= cfg.minimum_holding_time_s
                and lethality >= 1.0
            )
            slices.append(
                ThermalSlice(
                    take,
                    parcel.temp_c,
                    residence,
                    lethality,
                    thermal_safe,
                    fastest_residence_time_s=fastest_residence,
                    exponential_was_bounded=exponential_was_bounded,
                    mean_pass_count=parcel.mean_pass_count,
                    recycle_risk_fraction=parcel.recycle_risk_fraction,
                    chemical_fraction=parcel.chemical_fraction,
                    product_fraction=parcel.product_fraction,
                )
            )
            parcel.volume_l -= take
            parcel.entry_start_s = entry_cut
            remaining -= take
            outlet_consumed += take
            if parcel.volume_l <= _VOLUME_EPSILON_L:
                self.holding_queue.popleft()
        volume, temp, residence, lethality, safe_fraction = _weighted(slices)
        fastest_residence = _weighted_fastest_residence(slices)
        bounded_count = sum(item.exponential_was_bounded for item in slices)
        return HoldingOutlet(
            tuple(slices),
            volume,
            temp,
            residence,
            lethality,
            safe_fraction,
            fastest_residence,
            bounded_count,
        )

    def _move_to_fdv(self, inlet_slices: Iterable[ThermalSlice]) -> tuple[ThermalSlice, ...]:
        slices = list(inlet_slices)
        inlet_volume = sum(item.volume_l for item in slices)
        if inlet_volume <= 0.0:
            return ()
        for item in slices:
            self.fdv_line_queue.append(replace(item))
        remaining = inlet_volume
        outlet: list[ThermalSlice] = []
        while remaining > 1e-10:
            item = self.fdv_line_queue[0]
            take = min(remaining, item.volume_l)
            outlet.append(replace(item, volume_l=take))
            item.volume_l -= take
            remaining -= take
            if item.volume_l <= _VOLUME_EPSILON_L:
                self.fdv_line_queue.popleft()
        return tuple(outlet)

    def _move_post_fdv(
        self,
        inlet_slices: Iterable[ThermalSlice],
        effective_regeneration: float,
        cooler_factor: float,
        inlet_temp_c: float,
        downstream_fault_active: bool,
        process_differential_pressure_bar: float,
        pressure_safe: bool,
        contamination_risk: bool,
    ) -> tuple[ThermalSlice, ...]:
        """Displace the fixed regeneration/cooling inventory with forward liquid.

        Downstream attributes are attached on entry and therefore emerge only
        after the configured FIFO inventory has been displaced.
        """
        cfg = self.config
        slices = list(inlet_slices)
        inlet_volume_l = sum(item.volume_l for item in slices)
        if inlet_volume_l <= 0.0:
            return ()
        regeneration = min(1.0, max(0.0, effective_regeneration))
        cooling = min(1.0, max(0.0, cfg.cooler_effectiveness * cooler_factor))
        for item in slices:
            regen_hot_out_c = item.temp_c - regeneration * (
                item.temp_c - inlet_temp_c
            )
            product_temp_c = regen_hot_out_c - cooling * (
                regen_hot_out_c - cfg.final_product_target_c
            )
            if not math.isfinite(product_temp_c):
                raise FloatingPointError("post-FDV product temperature is not finite")
            self.post_fdv_queue.append(
                replace(
                    item,
                    product_temp_c=product_temp_c,
                    downstream_fault_active=downstream_fault_active,
                    downstream_regenerator_factor=regeneration,
                    downstream_cooler_factor=cooling,
                    process_differential_pressure_bar=(
                        process_differential_pressure_bar
                    ),
                    pressure_safe=pressure_safe,
                    contamination_risk=contamination_risk,
                )
            )

        remaining = inlet_volume_l
        outlet: list[ThermalSlice] = []
        while remaining > _VOLUME_EPSILON_L:
            item = self.post_fdv_queue[0]
            take = min(remaining, item.volume_l)
            outlet.append(replace(item, volume_l=take))
            item.volume_l -= take
            remaining -= take
            if item.volume_l <= _VOLUME_EPSILON_L:
                self.post_fdv_queue.popleft()
        return tuple(outlet)

    def _update_fdv(
        self,
        command_forward: bool,
        modifiers: ScenarioModifiers,
        step_dt_s: float,
        observed_safe_volume_l: float,
    ) -> tuple[float, float]:
        """Move the 0..1 valve and return an equivalent forward volume window."""
        cfg = self.config
        if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be finite and positive")
        if not math.isfinite(observed_safe_volume_l) or observed_safe_volume_l < 0.0:
            raise ValueError("observed_safe_volume_l must be finite and nonnegative")
        if (
            not math.isfinite(modifiers.valve_travel_time_factor)
            or modifiers.valve_travel_time_factor <= 0.0
        ):
            raise ValueError("valve_travel_time_factor must be finite and positive")
        if (
            not math.isfinite(modifiers.valve_leakage_fraction)
            or not 0.0 <= modifiers.valve_leakage_fraction <= 1.0
        ):
            raise ValueError("valve_leakage_fraction must be finite and in [0, 1]")
        position_start = min(1.0, max(0.0, self._fdv_position))
        if modifiers.valve_stuck_forward:
            self._fdv_position = 1.0
            self.forward_permissive_confirmed = bool(command_forward)
            self.pending_forward_safe_volume_l = 0.0
            self.pending_divert_time_s = 0.0
            return (0.0, 1.0)
        if not modifiers.power_available:
            self._fdv_position = 0.0
            self.forward_permissive_confirmed = False
            self.pending_forward_safe_volume_l = 0.0
            self.pending_divert_time_s = 0.0
            return (0.0, 0.0)

        leakage_position = min(1.0, max(0.0, modifiers.valve_leakage_fraction))
        gate_delay_s = 0.0
        if command_forward:
            self.pending_divert_time_s = 0.0
            if self.forward_permissive_confirmed:
                self.pending_forward_safe_volume_l = 0.0
                target_position = 1.0
            else:
                # The permissive is measured upstream of the FDV.  The valve
                # remains diverted until a continuous permissive volume has
                # displaced the line and confirmation volume.
                previous_safe_volume_l = self.pending_forward_safe_volume_l
                self.pending_forward_safe_volume_l += observed_safe_volume_l
                confirmation_volume_l = (
                    cfg.nominal_flow_l_s * cfg.forward_confirmation_s
                )
                required_safe_volume_l = cfg.fdv_line_volume_l + confirmation_volume_l
                if self.pending_forward_safe_volume_l + 1e-12 < required_safe_volume_l:
                    # A recovered upstream permissive does not prove that the
                    # sensor-to-FDV line has been flushed. Continue closing a
                    # partly open valve while confirmation volume accumulates.
                    target_position = leakage_position
                else:
                    needed_this_step_l = max(
                        0.0, required_safe_volume_l - previous_safe_volume_l
                    )
                    gate_fraction = (
                        needed_this_step_l / observed_safe_volume_l
                        if observed_safe_volume_l > 0.0
                        else 1.0
                    )
                    gate_delay_s = min(1.0, max(0.0, gate_fraction)) * step_dt_s
                    self.pending_forward_safe_volume_l = 0.0
                    self.forward_permissive_confirmed = True
                    target_position = 1.0
        else:
            self.pending_forward_safe_volume_l = 0.0
            self.forward_permissive_confirmed = False
            previous_divert_time_s = self.pending_divert_time_s
            self.pending_divert_time_s += step_dt_s
            if previous_divert_time_s < cfg.divert_actuation_delay_s:
                gate_delay_s = min(
                    step_dt_s,
                    cfg.divert_actuation_delay_s - previous_divert_time_s,
                )
            target_position = leakage_position

        travel_factor = max(1.0e-6, modifiers.valve_travel_time_factor)
        full_travel_time_s = cfg.fdv_travel_time_s * travel_factor
        motion_dt_s = max(0.0, step_dt_s - gate_delay_s)
        distance = target_position - position_start
        if abs(distance) <= 1.0e-15 or motion_dt_s <= 0.0:
            position_end = position_start
            position_integral_s = position_start * step_dt_s
        else:
            direction = 1.0 if distance > 0.0 else -1.0
            time_to_target_s = abs(distance) * full_travel_time_s
            moving_time_s = min(motion_dt_s, time_to_target_s)
            position_after_motion = position_start + direction * (
                moving_time_s / full_travel_time_s
            )
            if moving_time_s >= time_to_target_s - 1e-15:
                position_after_motion = target_position
            position_integral_s = position_start * gate_delay_s
            position_integral_s += (
                0.5 * (position_start + position_after_motion) * moving_time_s
            )
            position_integral_s += target_position * (motion_dt_s - moving_time_s)
            position_end = position_after_motion

        self._fdv_position = min(1.0, max(leakage_position, position_end))
        if not command_forward and abs(self._fdv_position - leakage_position) <= 1e-12:
            self.pending_divert_time_s = 0.0
        average_position = min(1.0, max(0.0, position_integral_s / step_dt_s))
        if average_position <= 1.0e-15:
            return (0.0, 0.0)
        if distance > 0.0:
            return (1.0 - average_position, 1.0)
        return (0.0, average_position)

    def _update_fouling(
        self,
        plant_mode: str,
        flow_ratio: float,
        process_temp_c: float,
        modifiers: ScenarioModifiers,
        step_dt_s: float,
    ) -> None:
        cfg = self.config
        if plant_mode.startswith("CIP_"):
            if plant_mode == "CIP_CAUSTIC":
                thermal_factor = min(1.0, max(0.0, (process_temp_c - 45.0) / 20.0))
                rate = (
                    cfg.cip_caustic_removal_rate_s
                    * thermal_factor
                    * modifiers.cleaning_effectiveness_factor
                )
                self._set_surface_from_fouling(
                    self.fouling_index * math.exp(-rate * step_dt_s)
                )
            elif plant_mode == "CIP_ACID":
                thermal_factor = min(1.0, max(0.0, (process_temp_c - 40.0) / 20.0))
                rate = (
                    cfg.cip_acid_removal_rate_s
                    * thermal_factor
                    * modifiers.cleaning_effectiveness_factor
                )
                self._set_surface_from_fouling(
                    self.fouling_index * math.exp(-rate * step_dt_s)
                )
            return
        if not self._surface_state_synchronized:
            self._set_surface_from_fouling(self.fouling_index)
        heat_factor = min(2.0, max(0.0, (process_temp_c - 55.0) / 19.0))
        growth = (
            cfg.fouling_rate_per_h
            / 3600.0
            * step_dt_s
            * modifiers.fouling_rate_factor
            * max(0.0, flow_ratio)
            * heat_factor
        )
        previous = self.fouling_index
        updated = min(1.0, previous + growth)
        deposited_total_g = (updated - previous) * self._surface_soil_scale_g
        if deposited_total_g > 0.0:
            configured_total = (
                cfg.cip_initial_protein_soil_g + cfg.cip_initial_mineral_soil_g
            )
            protein_share = cfg.cip_initial_protein_soil_g / configured_total
            self.cip_soil_model.deposit_soil(
                deposited_total_g * protein_share,
                deposited_total_g * (1.0 - protein_share),
            )
        self.fouling_index = updated
        self._surface_state_synchronized = True

    def _step_boundaries(
        self,
        scenario: str,
        duration_s: float | None = None,
    ) -> tuple[float, ...]:
        cfg = self.config
        duration = cfg.duration_s if duration_s is None else duration_s
        boundaries = {
            duration,
            cfg.fault_start_s,
            cfg.fault_start_s + cfg.fault_duration_s,
        }
        if scenario in {"cip_cycle", "incomplete_cleaning"}:
            boundaries.update({60.0, 240.0, 300.0, 420.0, 540.0})
        return tuple(sorted(value for value in boundaries if 0.0 < value <= duration))

    def _run_phase_records(
        self,
        scenario: str,
        *,
        duration_s: float | None,
        reset: bool,
        campaign_id: str | None,
        phase_id: str | None,
        phase_index: int,
    ) -> list[dict[str, float | int | str]]:
        if scenario not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario {scenario!r}")
        if reset:
            self.reset()
        cfg = self.config
        duration = cfg.duration_s if duration_s is None else float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("phase duration_s must be finite and positive")
        if math.ceil(duration / cfg.dt_s) > cfg.maximum_steps:
            raise ValueError("phase duration_s / dt_s exceeds maximum_steps")
        is_campaign = campaign_id is not None
        self._campaign_active = is_campaign
        campaign_offset_s = self._campaign_time_s if (is_campaign or not reset) else 0.0
        resolved_phase_id = phase_id or f"phase-{phase_index:03d}"
        phase_kind = (
            "cip"
            if scenario in {"cip_cycle", "incomplete_cleaning"}
            else "production"
        )
        if is_campaign:
            if not campaign_id or not campaign_id.strip():
                raise ValueError("campaign_id must be non-empty")
            if not resolved_phase_id.strip():
                raise ValueError("phase_id must be non-empty")
            if isinstance(phase_index, bool) or not isinstance(phase_index, int) or phase_index < 0:
                raise ValueError("phase_index must be a non-negative integer")
        if reset and scenario in {"cip_cycle", "incomplete_cleaning"}:
            self.cip_soil_model.reset()
            self.fouling_index = cfg.cip_initial_fouling_index
            self._surface_state_synchronized = True
        config_hash = _config_hash(cfg)
        run_id = (
            f"{scenario}-v{MODEL_VERSION}-{cfg.random_seed}-{config_hash[:12]}"
        )
        phase_run_id = (
            f"{campaign_id}-{phase_index:03d}-{resolved_phase_id}-{scenario}"
            if is_campaign
            else run_id
        )
        records: list[dict[str, float | int | str]] = []
        time_start_s = 0.0
        boundaries = self._step_boundaries(scenario, duration)
        while time_start_s < duration - 1e-12:
            if len(records) >= cfg.maximum_steps:
                raise RuntimeError(
                    f"simulation exceeded maximum_steps={cfg.maximum_steps}"
                )
            step_dt_s = min(cfg.dt_s, duration - time_start_s)
            for boundary in boundaries:
                if time_start_s + 1e-12 < boundary < time_start_s + step_dt_s - 1e-12:
                    step_dt_s = boundary - time_start_s
                    break
            time_end_s = time_start_s + step_dt_s
            absolute_time_start_s = campaign_offset_s + time_start_s
            absolute_time_end_s = campaign_offset_s + time_end_s
            sample_time_s = 0.5 * (time_start_s + time_end_s)
            modifiers = scenario_modifiers(scenario, sample_time_s, cfg)
            is_cip = scenario in {"cip_cycle", "incomplete_cleaning"}
            fault_active = int(
                scenario not in {"normal", "cip_cycle"}
                and _fault_active(cfg, sample_time_s)
            )

            if is_cip:
                scheduled_mode, inlet_temp_c, control_setpoint_c, cip_flow_factor, chemical_pct = cip_phase(sample_time_s)
                flow_l_s = cfg.nominal_flow_l_s * cip_flow_factor
                effective_regeneration = 0.0
            else:
                scheduled_mode = "PRODUCTION"
                inlet_temp_c = cfg.raw_milk_temp_c
                control_setpoint_c = cfg.pasteurization_setpoint_c
                chemical_pct = 0.0
                flow_l_s = cfg.nominal_flow_l_s * modifiers.flow_factor
                clean_regeneration = cfg.regenerator_effectiveness * modifiers.regenerator_factor
                effective_regeneration = clean_regeneration * (
                    1.0 - cfg.fouling_regeneration_loss_fraction * self.fouling_index
                )
                effective_regeneration *= int(self.last_routed_forward)
                effective_regeneration = min(0.98, max(0.0, effective_regeneration))

            parcel_volume_l = flow_l_s * step_dt_s
            flow_ratio = flow_l_s / cfg.nominal_flow_l_s if cfg.nominal_flow_l_s else 0.0
            fresh_feed_l = 0.0
            return_to_tank_l = 0.0
            inlet_mean_pass_count = 0.0
            inlet_recycle_risk_fraction = 0.0
            inlet_chemical_fraction = chemical_pct / 100.0 if is_cip else 0.0
            if is_campaign and is_cip:
                inlet_chemical_fraction = max(
                    inlet_chemical_fraction,
                    self.cip_soil_model.residual_chemical_fraction,
                )
            inlet_product_fraction = 0.0 if is_cip else 1.0
            tank_volume_balance_error_l = 0.0
            tank_temperature_moment_error_l_c = 0.0
            tank_pass_moment_error_l = 0.0
            tank_risk_volume_error_l = 0.0
            tank_chemical_volume_error_l = 0.0
            tank_product_volume_error_l = 0.0
            tank_ambient_heat_kj = 0.0
            transition_drained_l = 0.0
            transition_added_l = 0.0
            if not is_cip:
                returned = self.pending_return_stream
                return_to_tank_l = returned.volume_l
                fresh_feed_l = self.balance_tank.makeup_for_target(
                    cfg.balance_tank_initial_volume_l,
                    returned.volume_l,
                    parcel_volume_l,
                )
                if is_campaign and self._restart_guard_active:
                    transition_added_l = fresh_feed_l
                tank_step = self.balance_tank.step(
                    step_dt_s,
                    LiquidStream(fresh_feed_l, cfg.raw_milk_temp_c),
                    returned,
                    parcel_volume_l,
                )
                self.pending_return_stream = LiquidStream(
                    0.0, self.balance_tank.temperature_c
                )
                inlet_temp_c = tank_step.outlet.temperature_c
                inlet_mean_pass_count = tank_step.outlet.mean_pass_count + 1.0
                inlet_recycle_risk_fraction = tank_step.outlet.risk_fraction
                if is_campaign and not is_cip:
                    inlet_recycle_risk_fraction = max(
                        inlet_recycle_risk_fraction,
                        self._surface_hygiene_risk_fraction,
                    )
                inlet_chemical_fraction = tank_step.outlet.chemical_fraction
                inlet_product_fraction = tank_step.outlet.product_fraction
                tank_volume_balance_error_l = tank_step.volume_balance_error_l
                tank_temperature_moment_error_l_c = (
                    tank_step.temperature_moment_error_l_c
                )
                tank_pass_moment_error_l = tank_step.pass_moment_error_l
                tank_risk_volume_error_l = tank_step.risk_volume_error_l
                tank_chemical_volume_error_l = tank_step.chemical_volume_error_l
                tank_product_volume_error_l = tank_step.product_volume_error_l
                tank_ambient_heat_kj = tank_step.ambient_heat_kj
            elif is_campaign:
                # The production balance tank is isolated during this generic
                # CIP surrogate, but its thermal state continues to evolve.
                tank_step = self.balance_tank.step(
                    step_dt_s,
                    LiquidStream(0.0, cfg.raw_milk_temp_c),
                    LiquidStream(0.0, self.balance_tank.temperature_c),
                    0.0,
                )
                tank_volume_balance_error_l = tank_step.volume_balance_error_l
                tank_temperature_moment_error_l_c = (
                    tank_step.temperature_moment_error_l_c
                )
                tank_pass_moment_error_l = tank_step.pass_moment_error_l
                tank_risk_volume_error_l = tank_step.risk_volume_error_l
                tank_chemical_volume_error_l = tank_step.chemical_volume_error_l
                tank_product_volume_error_l = tank_step.product_volume_error_l
                tank_ambient_heat_kj = tank_step.ambient_heat_kj
                if self.pending_return_stream.volume_l > 0.0:
                    relax = 1.0 - math.exp(
                        -step_dt_s / cfg.balance_tank_heat_loss_tau_s
                    )
                    self.pending_return_stream = replace(
                        self.pending_return_stream,
                        temperature_c=(
                            self.pending_return_stream.temperature_c
                            + relax
                            * (
                                cfg.ambient_temp_c
                                - self.pending_return_stream.temperature_c
                            )
                        ),
                    )
            shadow_regenerator = self.regenerator_hx.step(
                step_dt_s,
                product_inlet_c=inlet_temp_c,
                product_flow_l_s=flow_l_s,
                utility_inlet_c=self.last_hold_temp_c,
                utility_flow_l_s=(
                    flow_l_s if self.last_routed_forward and not is_cip else 0.0
                ),
                fouling_index=self.fouling_index,
            )
            preheat_target = inlet_temp_c + effective_regeneration * (
                self.last_hold_temp_c - inlet_temp_c
            )
            if flow_l_s <= 0.0:
                preheat_target = cfg.ambient_temp_c
                self._relax_stationary_downstream(step_dt_s)
            regen_decay, _ = _bounded_exp(
                -step_dt_s / cfg.regenerator_tau_s,
                cfg.max_exponential_argument,
            )
            alpha_regen = 1.0 - regen_decay
            self.preheat_temp_c += (preheat_target - self.preheat_temp_c) * alpha_regen
            preheat_sensor_noise_c = self.rng.gauss(
                0.0, cfg.preheat_sensor_noise_std_c
            )
            if modifiers.sensor_available:
                preheat_sensor_c = self.preheat_temp_c + preheat_sensor_noise_c
                self.last_preheat_sensor_c = preheat_sensor_c
            else:
                preheat_sensor_c = self.last_preheat_sensor_c

            self.steam_valve = self._pi_control(
                self.last_control_sensor_c,
                preheat_sensor_c,
                control_setpoint_c,
                step_dt_s,
            )
            if not modifiers.power_available or flow_l_s <= 0.0:
                self.steam_valve = 0.0
            shadow_heater = self.heater_hx.step(
                step_dt_s,
                product_inlet_c=shadow_regenerator.product_outlet_c,
                product_flow_l_s=flow_l_s,
                utility_inlet_c=cfg.heating_utility_temp_c,
                utility_flow_l_s=(
                    cfg.nominal_flow_l_s
                    * self.steam_valve
                    * modifiers.heater_capacity_factor
                    if modifiers.power_available
                    else 0.0
                ),
                fouling_index=self.fouling_index,
            )
            fouling_heat_factor = 1.0 - cfg.fouling_heat_loss_fraction * self.fouling_index
            actual_capacity_gain = (
                modifiers.heater_capacity_factor
                * max(0.05, fouling_heat_factor)
                * cfg.nominal_flow_l_s
                / max(flow_l_s, cfg.nominal_flow_l_s * 0.05)
            )
            heater_target = self.preheat_temp_c + (
                self.steam_valve * cfg.heater_max_delta_c * actual_capacity_gain
            )
            if flow_l_s <= 0.0:
                heater_target = cfg.ambient_temp_c
            heater_decay, _ = _bounded_exp(
                -step_dt_s / cfg.heater_tau_s,
                cfg.max_exponential_argument,
            )
            alpha_heater = 1.0 - heater_decay
            self.heater_temp_c += (heater_target - self.heater_temp_c) * alpha_heater

            holding = self._move_through_holding_tube(
                parcel_volume_l,
                self.heater_temp_c,
                absolute_time_start_s,
                absolute_time_end_s,
                mean_pass_count=inlet_mean_pass_count,
                recycle_risk_fraction=inlet_recycle_risk_fraction,
                chemical_fraction=inlet_chemical_fraction,
                product_fraction=inlet_product_fraction,
            )
            hold_temp_c = holding.mean_temp_c
            control_sensor_noise_c = self.rng.gauss(
                0.0, cfg.control_sensor_noise_std_c
            )
            safety_sensor_noise_c = self.rng.gauss(
                0.0, cfg.safety_sensor_noise_std_c
            )
            if modifiers.sensor_available:
                control_sensor_c = (
                    hold_temp_c
                    + modifiers.control_sensor_bias_c
                    + control_sensor_noise_c
                )
                safety_sensor_c = (
                    hold_temp_c
                    + modifiers.safety_sensor_bias_c
                    + safety_sensor_noise_c
                )
                self.last_control_sensor_c = control_sensor_c
                self.last_safety_sensor_c = safety_sensor_c
            else:
                control_sensor_c = self.last_control_sensor_c
                safety_sensor_c = self.last_safety_sensor_c
            holding_chemical_fraction = _weighted_attribute(
                holding.slices, "chemical_fraction"
            )
            holding_product_fraction = _weighted_attribute(
                holding.slices, "product_fraction", 1.0
            )
            restart_product_interface_signal_fraction = holding_product_fraction
            if is_campaign and self._restart_guard_active and not is_cip:
                self._last_cip_conductivity_proxy_ms_cm = (
                    0.2 + 65.0 * holding_chemical_fraction
                )
                self._last_cip_ph_proxy = 7.0 + 5.0 * math.tanh(
                    80.0 * holding_chemical_fraction
                )
                self._cip_release_permissive = (
                    holding_chemical_fraction
                    <= cfg.cip_residual_chemical_threshold_fraction
                    and holding_product_fraction
                    >= 1.0 - cfg.cip_residual_chemical_threshold_fraction
                )
            cip_release_permissive_signal = int(
                not self._restart_guard_active or self._cip_release_permissive
            )
            measured_flow_l_h = max(
                1e-9,
                flow_l_s
                * 3600.0
                * (
                    1.0
                    + modifiers.flow_meter_bias_fraction
                    + self.rng.gauss(0.0, cfg.flow_meter_noise_std_fraction)
                ),
            )
            estimated_residence_s = (
                cfg.holding_tube_volume_l / (measured_flow_l_h / 3600.0)
                if flow_l_s > 0.0
                else 0.0
            )
            estimated_fastest_residence_s = (
                estimated_residence_s * cfg.fastest_flow_efficiency
            )

            raw_pressure_bar = cfg.raw_side_pressure_bar + 0.10 * (flow_ratio * flow_ratio - 1.0)
            pasteurized_pressure_bar = (
                raw_pressure_bar
                + cfg.booster_pressure_gain_bar * modifiers.booster_factor
                - cfg.pasteurized_line_drop_bar * flow_ratio * flow_ratio
                - cfg.fouling_pressure_drop_bar * self.fouling_index
            )
            actual_dp_bar = pasteurized_pressure_bar - raw_pressure_bar
            raw_pressure_sensor_bar = raw_pressure_bar + self.rng.gauss(
                0.0, cfg.pressure_sensor_noise_std_bar
            )
            pasteurized_pressure_sensor_bar = (
                pasteurized_pressure_bar
                + modifiers.pressure_sensor_bias_bar
                + self.rng.gauss(0.0, cfg.pressure_sensor_noise_std_bar)
            )
            measured_dp_bar = pasteurized_pressure_sensor_bar - raw_pressure_sensor_bar
            booster_pump_speed_fraction = max(
                0.0,
                modifiers.booster_factor
                + self.rng.gauss(0.0, cfg.booster_speed_sensor_noise_std_fraction),
            )
            leak_detector_signal_fraction = max(
                0.0,
                modifiers.leak_fraction
                + self.rng.gauss(0.0, cfg.leak_detector_noise_std_fraction),
            )
            sensor_disagreement_c = abs(control_sensor_c - safety_sensor_c)
            contamination_risk = int(
                modifiers.leak_fraction > 0.0
                and actual_dp_bar < cfg.required_pressure_differential_bar
            )
            pressure_safe_now = actual_dp_bar >= cfg.required_pressure_differential_bar
            holding_slices_at_sensor = tuple(
                replace(
                    item,
                    pressure_safe=pressure_safe_now,
                    contamination_risk=bool(contamination_risk),
                    process_differential_pressure_bar=actual_dp_bar,
                )
                for item in holding.slices
            )

            # CCP production alarms are inhibited during CIP.  CIP has its own
            # temperature/flow/conductivity/pH observables and completion gate;
            # evaluating milk-production limits during a cleaning recipe would
            # create semantically false PLC alarm events.
            alarm_low_temperature = int(
                not is_cip and safety_sensor_c < cfg.diversion_threshold_c
            )
            forward_temp_ok = (
                safety_sensor_c
                >= cfg.diversion_threshold_c + cfg.forward_temperature_margin_c
            )
            alarm_low_holding_time = int(
                not is_cip
                and estimated_fastest_residence_s < cfg.minimum_holding_time_s
            )
            alarm_high_flow = int(
                not is_cip and measured_flow_l_h > cfg.maximum_safe_flow_l_h
            )
            alarm_low_dp = int(
                not is_cip
                and measured_dp_bar < cfg.required_pressure_differential_bar
            )
            alarm_sensor_disagreement = int(
                sensor_disagreement_c > cfg.sensor_disagreement_limit_c
            )
            alarm_leak = int(
                leak_detector_signal_fraction >= cfg.leak_detector_alarm_fraction
            )
            # Explicit PLC/device-status inputs. The injected physical states
            # remain separate oracle fields in the generated artifacts.
            temperature_sensor_quality_ok = int(modifiers.sensor_available)
            power_good_signal = int(modifiers.power_available)
            cip_cycle_active = int(is_cip)
            alarm_sensor_dropout = int(not temperature_sensor_quality_ok)
            safety_permissive = (
                not is_cip
                and flow_l_s > 0.0
                and forward_temp_ok
                and not alarm_low_holding_time
                and not alarm_high_flow
                and not alarm_low_dp
                and not alarm_sensor_disagreement
                and not alarm_leak
                and not alarm_sensor_dropout
                and bool(power_good_signal)
                and (
                    not is_campaign
                    or not self._restart_guard_active
                    or bool(cip_release_permissive_signal)
                )
            )
            forward_interval = self._update_fdv(
                safety_permissive,
                modifiers,
                step_dt_s,
                parcel_volume_l,
            )
            feedback_noise = self.rng.gauss(
                0.0, cfg.fdv_position_sensor_noise_std_fraction
            )
            self.fdv_position_feedback = min(
                1.0,
                max(
                    0.0,
                    self._fdv_position
                    + modifiers.valve_feedback_bias_fraction
                    + feedback_noise,
                ),
            )
            commanded_position = float(
                safety_permissive
                and self.pending_forward_safe_volume_l <= _VOLUME_EPSILON_L
            )
            position_error = abs(self.fdv_position_feedback - commanded_position)
            if position_error > cfg.fdv_feedback_tolerance:
                self.fdv_mismatch_time_s += step_dt_s
            else:
                self.fdv_mismatch_time_s = 0.0
            expected_response_s = cfg.fdv_travel_time_s + (
                cfg.divert_actuation_delay_s if commanded_position <= 0.0 else 0.0
            )
            alarm_fdv_mismatch = int(
                position_error > cfg.fdv_feedback_tolerance
                and self.fdv_mismatch_time_s + 1e-12 >= expected_response_s
            )

            routed_slices = self._move_to_fdv(holding_slices_at_sensor)
            (
                routed_volume_l,
                routed_temp_c,
                routed_transport_residence_s,
                routed_lethality,
                routed_thermal_safe_fraction,
            ) = _weighted(routed_slices)
            routed_residence_s = _weighted_fastest_residence(routed_slices)
            routed_pressure_safe_fraction = (
                sum(item.volume_l for item in routed_slices if item.pressure_safe)
                / routed_volume_l
                if routed_volume_l > 0.0
                else 0.0
            )
            routed_contamination_risk_fraction = (
                sum(item.volume_l for item in routed_slices if item.contamination_risk)
                / routed_volume_l
                if routed_volume_l > 0.0
                else 0.0
            )
            routed_chemical_fraction = _weighted_attribute(
                routed_slices, "chemical_fraction"
            )
            routed_product_fraction = _weighted_attribute(
                routed_slices, "product_fraction", 1.0
            )
            routed_hygiene_risk_fraction = _weighted_attribute(
                routed_slices, "recycle_risk_fraction"
            )
            valve_forward_slices = (
                _slice_window(routed_slices, *forward_interval)
                if not is_cip
                else ()
            )
            diverted_slices = (
                _slice_window(routed_slices, 0.0, forward_interval[0])
                + _slice_window(routed_slices, forward_interval[1], 1.0)
                if not is_cip
                else ()
            )
            (
                valve_forward_l,
                valve_forward_temp_c,
                _,
                _,
                _,
            ) = _weighted(valve_forward_slices)
            shadow_cooler = self.cooler_hx.step(
                step_dt_s,
                product_inlet_c=(
                    valve_forward_temp_c
                    if valve_forward_l > 0.0
                    else self.last_hold_temp_c
                ),
                product_flow_l_s=valve_forward_l / step_dt_s,
                utility_inlet_c=cfg.cooling_utility_temp_c,
                utility_flow_l_s=(
                    cfg.nominal_flow_l_s * modifiers.cooler_factor
                    if valve_forward_l > 0.0 and modifiers.power_available
                    else 0.0
                ),
                fouling_index=self.fouling_index,
            )
            if valve_forward_l <= 0.0 and flow_l_s > 0.0:
                post_alpha = self._thermal_relaxation_alpha(step_dt_s)
                self._relax_slice_queue(self.post_fdv_queue, post_alpha)
            forward_slices = self._move_post_fdv(
                valve_forward_slices,
                effective_regeneration,
                modifiers.cooler_factor,
                inlet_temp_c,
                bool(fault_active),
                actual_dp_bar,
                pressure_safe_now,
                bool(contamination_risk),
            )
            transition_boundary_drained_l = 0.0
            if is_campaign and self._restart_guard_active and forward_slices:
                boundary_release_ready = all(
                    item.chemical_fraction
                    <= cfg.cip_residual_chemical_threshold_fraction
                    and item.product_fraction
                    >= 1.0 - cfg.cip_residual_chemical_threshold_fraction
                    and (
                        item.product_temp_c
                        if item.product_temp_c is not None
                        else item.temp_c
                    )
                    <= cfg.maximum_product_temp_c
                    for item in forward_slices
                )
                if boundary_release_ready:
                    self._restart_guard_active = False
                else:
                    transition_boundary_drained_l = sum(
                        (item.volume_l for item in forward_slices), 0.0
                    )
                    forward_slices = ()
                transition_drained_l += transition_boundary_drained_l
            (
                forward_l,
                forward_temp_c,
                forward_transport_residence_s,
                forward_lethality,
                forward_thermal_safe_fraction,
            ) = _weighted(forward_slices)
            forward_residence_s = _weighted_fastest_residence(forward_slices)
            forward_min_process_dp_bar = (
                min(
                    item.process_differential_pressure_bar
                    for item in forward_slices
                )
                if forward_slices
                else 0.0
            )
            forward_pressure_safe_fraction = (
                sum(item.volume_l for item in forward_slices if item.pressure_safe)
                / forward_l
                if forward_l > 0.0
                else 0.0
            )
            forward_contamination_risk_fraction = (
                sum(item.volume_l for item in forward_slices if item.contamination_risk)
                / forward_l
                if forward_l > 0.0
                else 0.0
            )
            forward_chemical_fraction = _weighted_attribute(
                forward_slices, "chemical_fraction"
            )
            forward_product_fraction = _weighted_attribute(
                forward_slices, "product_fraction", 1.0
            )
            forward_hygiene_risk_fraction = _weighted_attribute(
                forward_slices, "recycle_risk_fraction"
            )
            _, diverted_temp_c, _, _, _ = _weighted(diverted_slices)
            forward_thermal_safe_volume_l = sum(
                item.volume_l for item in forward_slices if item.thermal_safe
            )
            chemical_noncompliant_forward_l = sum(
                (
                    item.volume_l
                    for item in forward_slices
                    if item.chemical_fraction
                    > cfg.cip_residual_chemical_threshold_fraction
                ),
                0.0,
            )
            dilution_noncompliant_forward_l = sum(
                (
                    item.volume_l
                    for item in forward_slices
                    if item.product_fraction
                    < 1.0 - cfg.cip_residual_chemical_threshold_fraction
                ),
                0.0,
            )
            hygiene_noncompliant_forward_l = sum(
                (
                    item.volume_l
                    for item in forward_slices
                    if item.recycle_risk_fraction > 1e-12
                ),
                0.0,
            )
            forward_safe_volume_l = sum(
                item.volume_l
                for item in forward_slices
                if item.thermal_safe
                and item.pressure_safe
                and not item.contamination_risk
                and (
                    not is_campaign
                    or (
                        item.chemical_fraction
                        <= cfg.cip_residual_chemical_threshold_fraction
                        and item.product_fraction
                        >= 1.0 - cfg.cip_residual_chemical_threshold_fraction
                        and item.recycle_risk_fraction <= 1e-12
                    )
                )
            )
            unsafe_forward_l = max(0.0, forward_l - forward_safe_volume_l)
            diverted_l = routed_volume_l - valve_forward_l if not is_cip else 0.0
            cip_recirculated_l = routed_volume_l if is_cip else 0.0
            thermal_unsafe_forward_l = (
                max(0.0, forward_l - forward_thermal_safe_volume_l)
                if forward_l > 0.0
                else 0.0
            )
            pressure_noncompliant_forward_l = (
                sum(
                    (
                        item.volume_l
                        for item in forward_slices
                        if not item.pressure_safe
                    ),
                    0.0,
                )
            )
            contamination_exposed_forward_l = (
                sum(
                    (
                        item.volume_l
                        for item in forward_slices
                        if item.contamination_risk
                    ),
                    0.0,
                )
            )
            safe_forward_l = max(0.0, forward_l - unsafe_forward_l)
            routed_thermal_safe_volume_l = sum(
                item.volume_l for item in routed_slices if item.thermal_safe
            )
            routed_safe_volume_l = sum(
                item.volume_l
                for item in routed_slices
                if item.thermal_safe
                and item.pressure_safe
                and not item.contamination_risk
                and (
                    not is_campaign
                    or (
                        item.chemical_fraction
                        <= cfg.cip_residual_chemical_threshold_fraction
                        and item.product_fraction
                        >= 1.0 - cfg.cip_residual_chemical_threshold_fraction
                        and item.recycle_risk_fraction <= 1e-12
                    )
                )
            )
            routed_unsafe_volume_l = max(0.0, routed_volume_l - routed_safe_volume_l)
            actual_safe = int(routed_volume_l > 0.0 and routed_unsafe_volume_l <= 1e-9)

            potential_product_temp_c = _weighted_product_temp(
                forward_slices,
                self.last_product_temp_c,
            )
            if forward_l > 0.0:
                self.last_product_temp_c = potential_product_temp_c
            product_temp_sensor_c = potential_product_temp_c + self.rng.gauss(
                0.0, cfg.product_sensor_noise_std_c
            )
            return_temp_c = (
                routed_temp_c
                if is_cip
                else diverted_temp_c if diverted_l > 0.0 else 0.0
            )
            diverted_mean_pass_count = _weighted_attribute(
                diverted_slices, "mean_pass_count"
            )
            diverted_chemical_fraction = _weighted_attribute(
                diverted_slices, "chemical_fraction"
            )
            diverted_product_fraction = _weighted_attribute(
                diverted_slices, "product_fraction", 1.0
            )
            diverted_recycle_risk_fraction = (
                sum(
                    item.volume_l
                    * max(
                        item.recycle_risk_fraction,
                        1.0 if item.contamination_risk else 0.0,
                    )
                    for item in diverted_slices
                )
                / diverted_l
                if diverted_l > 0.0
                else 0.0
            )
            if not is_cip:
                if is_campaign and self._restart_guard_active:
                    # Restart flush is a once-through transition: returning
                    # rinse/product-interface fluid to the balance tank would
                    # dilute the tank and can create a non-terminating loop.
                    transition_drained_l += diverted_l
                    self.pending_return_stream = LiquidStream(
                        0.0,
                        diverted_temp_c
                        if diverted_l > 0.0
                        else cfg.raw_milk_temp_c,
                    )
                else:
                    self.pending_return_stream = LiquidStream(
                        diverted_l,
                        diverted_temp_c
                        if diverted_l > 0.0
                        else cfg.raw_milk_temp_c,
                        diverted_mean_pass_count,
                        diverted_recycle_risk_fraction,
                        diverted_chemical_fraction,
                        diverted_product_fraction,
                    )
            alarm_high_product_temp = int(
                not is_cip
                and product_temp_sensor_c > cfg.maximum_product_temp_c
            )
            quality_out_of_spec_l = (
                sum(
                    (
                        item.volume_l
                        for item in forward_slices
                        if (
                            (
                                item.product_temp_c
                                if item.product_temp_c is not None
                                else item.temp_c
                            )
                            > cfg.maximum_product_temp_c
                            or (
                                is_campaign
                                and (
                                    item.chemical_fraction
                                    > cfg.cip_residual_chemical_threshold_fraction
                                    or item.product_fraction
                                    < 1.0
                                    - cfg.cip_residual_chemical_threshold_fraction
                                    or item.recycle_risk_fraction > 1e-12
                                )
                            )
                        )
                    ),
                    0.0,
                )
            )
            product_downstream_fault_fraction = (
                sum(
                    item.volume_l
                    for item in forward_slices
                    if item.downstream_fault_active
                )
                / forward_l
                if forward_l > 0.0
                else 0.0
            )
            product_regenerator_factor = (
                sum(
                    item.volume_l * item.downstream_regenerator_factor
                    for item in forward_slices
                )
                / forward_l
                if forward_l > 0.0
                else 0.0
            )
            product_cooler_factor = (
                sum(
                    item.volume_l * item.downstream_cooler_factor
                    for item in forward_slices
                )
                / forward_l
                if forward_l > 0.0
                else 0.0
            )
            cip_effective_chemical_pct = (
                chemical_pct * modifiers.cleaning_effectiveness_factor
                if is_cip
                else 0.0
            )
            cip_step: CIPStep | None = None
            if is_cip:
                cip_step = self.cip_soil_model.step(
                    step_dt_s,
                    temperature_c=self.heater_temp_c,
                    velocity_m_s=cfg.cip_reference_velocity_m_s * flow_ratio,
                    alkali_pct=(
                        cip_effective_chemical_pct
                        if scheduled_mode == "CIP_CAUSTIC"
                        else 0.0
                    ),
                    acid_pct=(
                        cip_effective_chemical_pct
                        if scheduled_mode == "CIP_ACID"
                        else 0.0
                    ),
                    rinse_flow_l_s=flow_l_s,
                )
                self._set_fouling_from_surface()
                self._last_cip_conductivity_proxy_ms_cm = (
                    cip_step.conductivity_proxy_ms_cm
                )
                self._last_cip_ph_proxy = cip_step.ph_proxy
                conductivity_limit = 0.2 + 65.0 * (
                    cfg.cip_residual_chemical_threshold_fraction
                )
                self._cip_release_permissive = (
                    cip_step.conductivity_proxy_ms_cm <= conductivity_limit
                    and 6.5 <= cip_step.ph_proxy <= 8.0
                )
                cip_release_permissive_signal = int(
                    self._cip_release_permissive
                )
            else:
                self._update_fouling(
                    "PRODUCTION",
                    flow_ratio,
                    self.heater_temp_c,
                    modifiers,
                    step_dt_s,
                )
            alarm_high_fouling = int(self.fouling_index >= cfg.fouling_alarm_index)
            alarm_unsafe_forward = int(unsafe_forward_l > 1e-9)
            alarm_count = sum(
                (
                    alarm_low_temperature,
                    alarm_low_holding_time,
                    alarm_high_flow,
                    alarm_low_dp,
                    alarm_sensor_disagreement,
                    alarm_leak,
                    alarm_fdv_mismatch,
                    alarm_high_product_temp,
                    alarm_sensor_dropout,
                )
            )
            observable_alarm_count = alarm_count
            diagnostic_alarm_count = alarm_unsafe_forward + alarm_high_fouling
            bounded_exponential_diagnostic = int(
                holding.bounded_exponential_count > 0
            )
            diagnostic_count = (
                diagnostic_alarm_count + bounded_exponential_diagnostic
            )

            if is_cip:
                plant_mode = scheduled_mode
            elif forward_l > 0.0:
                plant_mode = "PRODUCTION"
                self.ever_forward = True
            elif safety_permissive:
                plant_mode = "RECOVERY"
            elif self.ever_forward:
                plant_mode = "DIVERT"
            else:
                plant_mode = "STARTUP"

            mass_flow_kg_s = flow_l_s * cfg.density_kg_l
            heat_kw = max(
                0.0,
                mass_flow_kg_s
                * cfg.heat_capacity_kj_kg_k
                * (self.heater_temp_c - self.preheat_temp_c),
            )
            regeneration_saved_kw = (
                max(
                    0.0,
                    mass_flow_kg_s
                    * cfg.heat_capacity_kj_kg_k
                    * (self.preheat_temp_c - inlet_temp_c),
                )
                if self.last_routed_forward
                else 0.0
            )
            flow_m3_s = flow_l_s / 1000.0
            booster_delta_pa = max(0.0, pasteurized_pressure_bar - raw_pressure_bar) * 100_000.0
            pump_power_kw = flow_m3_s * booster_delta_pa / cfg.pump_efficiency / 1000.0
            heat_energy_kwh = heat_kw * step_dt_s / 3600.0
            pump_energy_kwh = pump_power_kw * step_dt_s / 3600.0
            mass_balance_error_l = (
                routed_volume_l
                - valve_forward_l
                - diverted_l
                - cip_recirculated_l
            )
            post_fdv_balance_error_l = (
                valve_forward_l - forward_l - transition_boundary_drained_l
            )

            holding_inventory_l = sum(parcel.volume_l for parcel in self.holding_queue)
            fdv_line_inventory_l = sum(item.volume_l for item in self.fdv_line_queue)
            post_fdv_inventory_l = sum(item.volume_l for item in self.post_fdv_queue)

            row: dict[str, float | int | str] = {
                    "model_version": MODEL_VERSION,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "random_seed": cfg.random_seed,
                    "scenario": scenario,
                    "time_start_s": round(time_start_s, 9),
                    "time_s": round(time_end_s, 9),
                    "step_dt_s": step_dt_s,
                    "fault_active": fault_active,
                    "plant_mode": plant_mode,
                    "flow_l_h": flow_l_s * 3600.0,
                    "measured_flow_l_h": measured_flow_l_h,
                    "maximum_safe_flow_l_h": cfg.maximum_safe_flow_l_h,
                    "inlet_temp_c": inlet_temp_c,
                    "fresh_feed_l": fresh_feed_l,
                    "return_to_balance_tank_l": return_to_tank_l,
                    "return_line_inventory_l": self.pending_return_stream.volume_l,
                    "balance_tank_volume_l": self.balance_tank.volume_l,
                    "balance_tank_temp_c": self.balance_tank.temperature_c,
                    "balance_tank_mean_pass_count": self.balance_tank.mean_pass_count,
                    "balance_tank_risk_fraction": self.balance_tank.risk_fraction,
                    "balance_tank_chemical_fraction": self.balance_tank.chemical_fraction,
                    "balance_tank_product_fraction": self.balance_tank.product_fraction,
                    "inlet_mean_pass_count": inlet_mean_pass_count,
                    "inlet_recycle_risk_fraction": inlet_recycle_risk_fraction,
                    "inlet_chemical_fraction": inlet_chemical_fraction,
                    "inlet_product_fraction": inlet_product_fraction,
                    "preheat_temp_c": self.preheat_temp_c,
                    "preheat_temp_sensor_c": preheat_sensor_c,
                    "heater_out_temp_c": self.heater_temp_c,
                    "shadow_regenerator_cold_out_c": (
                        shadow_regenerator.product_outlet_c
                    ),
                    "shadow_regenerator_hot_out_c": (
                        shadow_regenerator.utility_outlet_c
                    ),
                    "shadow_regenerator_wall_temp_c": (
                        shadow_regenerator.wall_temperature_c
                    ),
                    "shadow_regenerator_effective_ua_kw_k": (
                        shadow_regenerator.effective_ua_kw_k
                    ),
                    "shadow_regenerator_energy_balance_error_kj": (
                        shadow_regenerator.energy_balance_error_kj
                    ),
                    "shadow_heater_product_out_c": shadow_heater.product_outlet_c,
                    "shadow_heater_utility_out_c": shadow_heater.utility_outlet_c,
                    "shadow_heater_wall_temp_c": shadow_heater.wall_temperature_c,
                    "shadow_heater_effective_ua_kw_k": shadow_heater.effective_ua_kw_k,
                    "shadow_heater_energy_balance_error_kj": (
                        shadow_heater.energy_balance_error_kj
                    ),
                    "holding_out_temp_c": hold_temp_c,
                    "control_temp_sensor_c": control_sensor_c,
                    "safety_temp_sensor_c": safety_sensor_c,
                    "sensor_disagreement_c": sensor_disagreement_c,
                    "actual_residence_time_s": holding.mean_residence_time_s,
                    "fastest_residence_time_s": holding.mean_fastest_residence_time_s,
                    "mean_transport_residence_time_s": holding.mean_residence_time_s,
                    "fastest_flow_efficiency": cfg.fastest_flow_efficiency,
                    "estimated_residence_time_s": estimated_residence_s,
                    "estimated_fastest_residence_time_s": estimated_fastest_residence_s,
                    "relative_lethality": holding.mean_relative_lethality,
                    "outlet_thermal_safe_fraction": holding.thermal_safe_fraction,
                    "holding_chemical_fraction": holding_chemical_fraction,
                    "holding_product_fraction": holding_product_fraction,
                    "routed_temp_c": routed_temp_c,
                    "routed_residence_time_s": routed_transport_residence_s,
                    "routed_fastest_residence_time_s": routed_residence_s,
                    "routed_transport_residence_time_s": routed_transport_residence_s,
                    "routed_relative_lethality": routed_lethality,
                    "routed_thermal_safe_fraction": routed_thermal_safe_fraction,
                    "routed_pressure_safe_fraction": routed_pressure_safe_fraction,
                    "routed_contamination_risk_fraction": routed_contamination_risk_fraction,
                    "routed_chemical_fraction": routed_chemical_fraction,
                    "routed_product_fraction": routed_product_fraction,
                    "routed_hygiene_risk_fraction": routed_hygiene_risk_fraction,
                    "forward_temp_c": forward_temp_c,
                    "forward_residence_time_s": forward_transport_residence_s,
                    "forward_fastest_residence_time_s": forward_residence_s,
                    "forward_transport_residence_time_s": forward_transport_residence_s,
                    "forward_relative_lethality": forward_lethality,
                    "forward_thermal_safe_fraction": forward_thermal_safe_fraction,
                    "forward_min_process_differential_pressure_bar": forward_min_process_dp_bar,
                    "forward_pressure_safe_fraction": forward_pressure_safe_fraction,
                    "forward_contamination_risk_fraction": forward_contamination_risk_fraction,
                    "forward_chemical_fraction": forward_chemical_fraction,
                    "forward_product_fraction": forward_product_fraction,
                    "forward_hygiene_risk_fraction": forward_hygiene_risk_fraction,
                    "steam_valve": self.steam_valve,
                    "raw_pressure_bar": raw_pressure_bar,
                    "pasteurized_pressure_bar": pasteurized_pressure_bar,
                    "raw_pressure_sensor_bar": raw_pressure_sensor_bar,
                    "pasteurized_pressure_sensor_bar": pasteurized_pressure_sensor_bar,
                    "differential_pressure_bar": actual_dp_bar,
                    "measured_differential_pressure_bar": measured_dp_bar,
                    "booster_pump_factor": modifiers.booster_factor,
                    "booster_pump_speed_fraction": booster_pump_speed_fraction,
                    "leak_fraction": modifiers.leak_fraction,
                    "leak_detector_signal_fraction": leak_detector_signal_fraction,
                    "contamination_risk": contamination_risk,
                    "fouling_index": self.fouling_index,
                    "surface_protein_soil_g": self.cip_soil_model.protein_soil_g,
                    "surface_mineral_soil_g": self.cip_soil_model.mineral_soil_g,
                    "surface_total_soil_g": self.cip_soil_model.total_soil_g,
                    "surface_hygiene_risk_fraction": (
                        self._surface_hygiene_risk_fraction
                    ),
                    "cip_chemical_concentration_pct": chemical_pct,
                    "cip_effective_chemical_concentration_pct": (
                        cip_effective_chemical_pct
                    ),
                    "cip_protein_soil_g": (
                        cip_step.protein_soil_g if cip_step is not None else 0.0
                    ),
                    "cip_mineral_soil_g": (
                        cip_step.mineral_soil_g if cip_step is not None else 0.0
                    ),
                    "cip_total_soil_g": (
                        cip_step.total_soil_g if cip_step is not None else 0.0
                    ),
                    "cip_removed_protein_g": (
                        cip_step.removed_protein_g if cip_step is not None else 0.0
                    ),
                    "cip_removed_mineral_g": (
                        cip_step.removed_mineral_g if cip_step is not None else 0.0
                    ),
                    "cip_residual_chemical_fraction": (
                        cip_step.residual_chemical_fraction
                        if cip_step is not None
                        else 0.0
                    ),
                    "cip_conductivity_proxy_ms_cm": (
                        cip_step.conductivity_proxy_ms_cm
                        if cip_step is not None
                        else 0.2
                    ),
                    "cip_ph_proxy": cip_step.ph_proxy if cip_step is not None else 7.0,
                    "cip_cycle_active": cip_cycle_active,
                    "cip_release_permissive": cip_release_permissive_signal,
                    "post_cip_conductivity_proxy_ms_cm": (
                        self._last_cip_conductivity_proxy_ms_cm
                    ),
                    "post_cip_ph_proxy": self._last_cip_ph_proxy,
                    "restart_product_interface_signal_fraction": (
                        restart_product_interface_signal_fraction
                    ),
                    "cip_cleaning_complete": int(
                        cip_step.cleaning_complete if cip_step is not None else False
                    ),
                    "fdv_command_forward": int(safety_permissive),
                    "fdv_command_position": commanded_position,
                    "fdv_actual_forward": int(
                        self._fdv_position > cfg.fdv_feedback_tolerance
                    ),
                    "fdv_position": self._fdv_position,
                    "fdv_position_feedback": self.fdv_position_feedback,
                    "fdv_position_error": position_error,
                    "fdv_mismatch_time_s": self.fdv_mismatch_time_s,
                    "fdv_forward_fraction": (
                        valve_forward_l / routed_volume_l if routed_volume_l > 0.0 else 0.0
                    ),
                    "actual_safe": actual_safe,
                    "routed_volume_l": routed_volume_l,
                    "forward_l": forward_l,
                    "valve_forward_l": valve_forward_l,
                    "post_fdv_inlet_l": valve_forward_l,
                    "product_boundary_l": forward_l,
                    "safe_forward_l": safe_forward_l,
                    "diverted_l": diverted_l,
                    "cip_recirculated_l": cip_recirculated_l,
                    "unsafe_forward_l": unsafe_forward_l,
                    "thermal_unsafe_forward_l": thermal_unsafe_forward_l,
                    "pressure_noncompliant_forward_l": pressure_noncompliant_forward_l,
                    "contamination_exposed_forward_l": contamination_exposed_forward_l,
                    "chemical_noncompliant_forward_l": (
                        chemical_noncompliant_forward_l
                    ),
                    "dilution_noncompliant_forward_l": (
                        dilution_noncompliant_forward_l
                    ),
                    "hygiene_noncompliant_forward_l": (
                        hygiene_noncompliant_forward_l
                    ),
                    "transition_drained_l": transition_drained_l,
                    "transition_added_l": transition_added_l,
                    "quality_out_of_spec_l": quality_out_of_spec_l,
                    "product_temp_c": potential_product_temp_c,
                    "product_temp_sensor_c": product_temp_sensor_c,
                    "potential_product_temp_c": potential_product_temp_c,
                    "shadow_cooler_product_out_c": shadow_cooler.product_outlet_c,
                    "shadow_cooler_utility_out_c": shadow_cooler.utility_outlet_c,
                    "shadow_cooler_wall_temp_c": shadow_cooler.wall_temperature_c,
                    "shadow_cooler_effective_ua_kw_k": shadow_cooler.effective_ua_kw_k,
                    "shadow_cooler_energy_balance_error_kj": (
                        shadow_cooler.energy_balance_error_kj
                    ),
                    "product_downstream_fault_fraction": product_downstream_fault_fraction,
                    "product_regenerator_factor": product_regenerator_factor,
                    "product_cooler_factor": product_cooler_factor,
                    "return_temp_c": return_temp_c,
                    "diverted_mean_pass_count": diverted_mean_pass_count,
                    "diverted_recycle_risk_fraction": (
                        diverted_recycle_risk_fraction
                    ),
                    "diverted_chemical_fraction": diverted_chemical_fraction,
                    "diverted_product_fraction": diverted_product_fraction,
                    "heat_kw": heat_kw,
                    "regeneration_saved_kw": regeneration_saved_kw,
                    "pump_power_kw": pump_power_kw,
                    "heat_energy_kwh": heat_energy_kwh,
                    "pump_energy_kwh": pump_energy_kwh,
                    "holding_inventory_l": holding_inventory_l,
                    "fdv_line_inventory_l": fdv_line_inventory_l,
                    "post_fdv_inventory_l": post_fdv_inventory_l,
                    "mass_balance_error_l": mass_balance_error_l,
                    "post_fdv_balance_error_l": post_fdv_balance_error_l,
                    "balance_tank_volume_error_l": tank_volume_balance_error_l,
                    "balance_tank_temperature_moment_error_l_c": (
                        tank_temperature_moment_error_l_c
                    ),
                    "balance_tank_pass_moment_error_l": (
                        tank_pass_moment_error_l
                    ),
                    "balance_tank_risk_volume_error_l": tank_risk_volume_error_l,
                    "balance_tank_chemical_volume_error_l": (
                        tank_chemical_volume_error_l
                    ),
                    "balance_tank_product_volume_error_l": (
                        tank_product_volume_error_l
                    ),
                    "balance_tank_ambient_heat_kj": tank_ambient_heat_kj,
                    "alarm_low_temperature": alarm_low_temperature,
                    "alarm_low_holding_time": alarm_low_holding_time,
                    "alarm_high_flow": alarm_high_flow,
                    "alarm_low_pressure_differential": alarm_low_dp,
                    "alarm_sensor_disagreement": alarm_sensor_disagreement,
                    "alarm_regenerator_leak": alarm_leak,
                    "alarm_fdv_mismatch": alarm_fdv_mismatch,
                    "alarm_high_product_temperature": alarm_high_product_temp,
                    "alarm_high_fouling": alarm_high_fouling,
                    "alarm_sensor_dropout": alarm_sensor_dropout,
                    "alarm_unsafe_forward": alarm_unsafe_forward,
                    "alarm_count": alarm_count,
                    "observable_alarm_count": observable_alarm_count,
                    "diagnostic_alarm_count": diagnostic_alarm_count,
                    "oracle_alarm_count": diagnostic_alarm_count,
                    "bounded_exponential_count": holding.bounded_exponential_count,
                    "bounded_exponential_diagnostic": bounded_exponential_diagnostic,
                    "diagnostic_count": diagnostic_count,
                    "total_alarm_count": alarm_count + diagnostic_alarm_count,
                    "control_sensor_bias_c": modifiers.control_sensor_bias_c,
                    "safety_sensor_bias_c": modifiers.safety_sensor_bias_c,
                    "flow_meter_bias_fraction": modifiers.flow_meter_bias_fraction,
                    "heater_capacity_factor": modifiers.heater_capacity_factor,
                    "regenerator_factor": modifiers.regenerator_factor,
                    "cooler_factor": modifiers.cooler_factor,
                    "power_available": int(modifiers.power_available),
                    "power_good_signal": power_good_signal,
                    "sensor_available": int(modifiers.sensor_available),
                    "temperature_sensor_quality_ok": (
                        temperature_sensor_quality_ok
                    ),
                    "valve_travel_time_factor": modifiers.valve_travel_time_factor,
                    "valve_leakage_fraction": modifiers.valve_leakage_fraction,
                    "cleaning_effectiveness_factor": modifiers.cleaning_effectiveness_factor,
                }
            if is_campaign:
                row.update(
                    {
                        "campaign_id": campaign_id or "",
                        "phase_id": resolved_phase_id,
                        "phase_index": phase_index,
                        "phase_kind": phase_kind,
                        "phase_run_id": phase_run_id,
                        "campaign_time_start_s": round(
                            absolute_time_start_s, 9
                        ),
                        "campaign_time_s": round(absolute_time_end_s, 9),
                    }
                )
            _ensure_finite_mapping(row, f"record[{len(records)}]")
            records.append(row)
            self.last_hold_temp_c = hold_temp_c
            self.last_control_sensor_c = control_sensor_c
            self.last_routed_forward = valve_forward_l > 0.0
            time_start_s = time_end_s
        self._campaign_time_s = campaign_offset_s + duration
        return records

    @staticmethod
    def _scenario_phase_kind(scenario: str) -> str:
        return "cip" if scenario in {"cip_cycle", "incomplete_cleaning"} else "production"

    def _prepare_phase_transition(self, next_kind: str) -> None:
        """Apply the explicit control transition while retaining physical state."""
        if next_kind not in {"production", "cip"}:
            raise ValueError(f"unsupported phase kind {next_kind!r}")
        previous_kind = self._last_phase_kind
        if previous_kind == "cip" and next_kind == "production":
            soil = self.cip_soil_model.total_soil_g
            threshold = self.config.cip_clean_soil_threshold_g
            self._surface_hygiene_risk_fraction = min(
                1.0,
                max(0.0, soil - threshold) / max(soil, threshold, 1e-12),
            )
            self._restart_guard_active = True
        elif next_kind == "cip":
            self._restart_guard_active = False
            self._cip_release_permissive = False
            self._surface_hygiene_risk_fraction = 0.0

        # These are controller/phase memories, not physical inventories.  The
        # actual FDV position, queues, tank, heat exchangers and RNG carry over.
        self.controller_integral = 0.0
        self.steam_valve = 0.0
        self.forward_permissive_confirmed = False
        self.pending_forward_safe_volume_l = 0.0
        self.pending_divert_time_s = 0.0
        self.fdv_mismatch_time_s = 0.0
        self.ever_forward = False
        self._last_phase_kind = next_kind

    def run(self, scenario: str = "normal") -> list[dict[str, float | int | str]]:
        """Run one backward-compatible, independently reset scenario."""
        return self._run_phase_records(
            scenario,
            duration_s=None,
            reset=True,
            campaign_id=None,
            phase_id=None,
            phase_index=0,
        )

    def run_phase(
        self,
        scenario: str,
        *,
        duration_s: float | None = None,
        reset: bool = False,
        campaign_id: str | None = None,
        phase_id: str | None = None,
        phase_index: int = 0,
    ) -> list[dict[str, float | int | str]]:
        """Run one phase, optionally retaining the complete current state."""
        if campaign_id is None:
            return self._run_phase_records(
                scenario,
                duration_s=duration_s,
                reset=reset,
                campaign_id=None,
                phase_id=phase_id,
                phase_index=phase_index,
            )
        if reset:
            self.reset()
        self._campaign_active = True
        self._prepare_phase_transition(self._scenario_phase_kind(scenario))
        return self._run_phase_records(
            scenario,
            duration_s=duration_s,
            reset=False,
            campaign_id=campaign_id,
            phase_id=phase_id,
            phase_index=phase_index,
        )

    def run_campaign(
        self,
        phases: Sequence[CampaignPhase | Mapping[str, object]],
        *,
        campaign_id: str = "campaign",
        reset: bool = True,
        initial_state: Mapping[str, object] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Run ordered phases on one continuous physical and stochastic state."""
        if not campaign_id or not campaign_id.strip():
            raise ValueError("campaign_id must be non-empty")
        normalized: list[CampaignPhase] = []
        for raw in phases:
            if isinstance(raw, CampaignPhase):
                phase = raw
            elif isinstance(raw, Mapping):
                if set(raw) != {"phase_id", "scenario", "duration_s"}:
                    raise ValueError("campaign phase mapping has invalid fields")
                phase = CampaignPhase(
                    phase_id=str(raw["phase_id"]),
                    scenario=str(raw["scenario"]),
                    duration_s=float(raw["duration_s"]),
                )
            else:
                raise ValueError("campaign phases must be CampaignPhase or mappings")
            normalized.append(phase)
        if not normalized:
            raise ValueError("campaign requires at least one phase")
        phase_ids = [phase.phase_id for phase in normalized]
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("campaign phase_id values must be unique")
        if initial_state is not None:
            self.import_state(initial_state)
        elif reset:
            self.reset()
        self._campaign_active = True

        records: list[dict[str, float | int | str]] = []
        for index, phase in enumerate(normalized):
            self._prepare_phase_transition(
                self._scenario_phase_kind(phase.scenario)
            )
            records.extend(
                self._run_phase_records(
                    phase.scenario,
                    duration_s=phase.duration_s,
                    reset=False,
                    campaign_id=campaign_id,
                    phase_id=phase.phase_id,
                    phase_index=index,
                )
            )
        return records


def extract_events(
    records: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    events: list[dict[str, float | str]] = []
    previous_alarms = {name: 0 for name in ALARM_COLUMNS}
    previous_mode: str | None = None
    for row in records:
        # Row states apply over [time_start_s, time_s].  Timestamp edges at
        # the interval start so configured fault/CIP boundaries are not
        # reported one sampling period late.  Synthetic callers without the
        # interval field retain the historical end-time behaviour.
        time_s = float(row.get("time_start_s", row["time_s"]))
        mode = str(row["plant_mode"])
        if mode != previous_mode:
            events.append(
                {
                    "time_s": time_s,
                    "event_type": "MODE",
                    "event_name": mode,
                    "state": "ENTERED",
                }
            )
            previous_mode = mode
        for alarm in ALARM_COLUMNS:
            state = int(row.get(alarm, 0))
            if state != previous_alarms[alarm]:
                events.append(
                    {
                        "time_s": time_s,
                        "event_type": "ALARM",
                        "event_name": alarm,
                        "state": "ACTIVE" if state else "CLEARED",
                    }
                )
                previous_alarms[alarm] = state
    return events


def summarize(
    records: list[dict[str, float | int | str]],
    config: HTSTConfig,
) -> dict[str, object]:
    if not records:
        raise ValueError("Cannot summarize an empty simulation")
    total_routed_l = sum(float(row["routed_volume_l"]) for row in records)
    total_forward_l = sum(float(row["forward_l"]) for row in records)
    total_diverted_l = sum(float(row["diverted_l"]) for row in records)
    total_cip_l = sum(float(row["cip_recirculated_l"]) for row in records)
    unsafe_forward_l = sum(float(row["unsafe_forward_l"]) for row in records)
    thermal_unsafe_forward_l = sum(
        float(row["thermal_unsafe_forward_l"]) for row in records
    )
    pressure_noncompliant_forward_l = sum(
        float(row["pressure_noncompliant_forward_l"]) for row in records
    )
    contamination_exposed_forward_l = sum(
        float(row["contamination_exposed_forward_l"]) for row in records
    )
    chemical_noncompliant_forward_l = sum(
        float(row["chemical_noncompliant_forward_l"]) for row in records
    )
    dilution_noncompliant_forward_l = sum(
        float(row["dilution_noncompliant_forward_l"]) for row in records
    )
    hygiene_noncompliant_forward_l = sum(
        float(row["hygiene_noncompliant_forward_l"]) for row in records
    )
    transition_drained_l = sum(
        float(row["transition_drained_l"]) for row in records
    )
    transition_added_l = sum(
        float(row["transition_added_l"]) for row in records
    )
    quality_oos_l = sum(float(row["quality_out_of_spec_l"]) for row in records)
    forward_rows = [row for row in records if float(row["forward_l"]) > 0.0]
    first_forward = next(
        (float(row["time_s"]) for row in forward_rows),
        None,
    )
    state_durations: dict[str, float] = collections.defaultdict(float)
    alarm_durations: dict[str, float] = collections.defaultdict(float)
    for row in records:
        dt = float(row["step_dt_s"])
        state_durations[str(row["plant_mode"])] += dt
        for alarm in ALARM_COLUMNS:
            if int(row[alarm]):
                alarm_durations[alarm] += dt
    events = extract_events(records)
    thermal_energy = sum(float(row["heat_energy_kwh"]) for row in records)
    pump_energy = sum(float(row["pump_energy_kwh"]) for row in records)
    initial_fouling = (
        config.cip_initial_fouling_index
        if records[0]["scenario"] in {"cip_cycle", "incomplete_cleaning"}
        else config.initial_fouling_index
    )
    final_fouling = float(records[-1]["fouling_index"])
    total_fresh_feed_l = sum(float(row["fresh_feed_l"]) for row in records)
    total_return_to_tank_l = sum(
        float(row["return_to_balance_tank_l"]) for row in records
    )
    final_tank_volume_l = float(records[-1]["balance_tank_volume_l"])
    final_return_line_l = float(records[-1]["return_line_inventory_l"])
    external_volume_balance_error_l = (
        total_fresh_feed_l
        - total_forward_l
        - transition_drained_l
        - (final_tank_volume_l - config.balance_tank_initial_volume_l)
        - final_return_line_l
    )
    return {
        "model_version": MODEL_VERSION,
        "run_id": records[0]["run_id"],
        "config_hash": records[0]["config_hash"],
        "scenario": records[0]["scenario"],
        "config": asdict(config),
        "duration_s": float(records[-1]["time_s"]),
        "routed_l": round(total_routed_l, 3),
        "forward_l": round(total_forward_l, 3),
        "diverted_l": round(total_diverted_l, 3),
        "cip_recirculated_l": round(total_cip_l, 3),
        "cip_processed_l": round(total_cip_l, 3),
        "fresh_feed_l": round(total_fresh_feed_l, 3),
        "returned_to_balance_tank_l": round(total_return_to_tank_l, 3),
        "transition_drained_l": round(transition_drained_l, 6),
        "transition_added_l": round(transition_added_l, 6),
        "final_balance_tank_volume_l": round(final_tank_volume_l, 6),
        "final_return_line_inventory_l": round(final_return_line_l, 6),
        "external_volume_balance_error_l": round(
            external_volume_balance_error_l, 9
        ),
        "diverted_fraction": round(total_diverted_l / total_routed_l, 6)
        if total_routed_l
        else 0.0,
        "unsafe_forward_l": round(unsafe_forward_l, 6),
        "thermal_unsafe_forward_l": round(thermal_unsafe_forward_l, 6),
        "pressure_noncompliant_forward_l": round(
            pressure_noncompliant_forward_l, 6
        ),
        "contamination_exposed_forward_l": round(
            contamination_exposed_forward_l, 6
        ),
        "chemical_noncompliant_forward_l": round(
            chemical_noncompliant_forward_l, 6
        ),
        "dilution_noncompliant_forward_l": round(
            dilution_noncompliant_forward_l, 6
        ),
        "hygiene_noncompliant_forward_l": round(
            hygiene_noncompliant_forward_l, 6
        ),
        "unsafe_forward_fraction_of_forward": round(
            unsafe_forward_l / total_forward_l, 6
        )
        if total_forward_l
        else None,
        "quality_out_of_spec_l": round(quality_oos_l, 6),
        "first_forward_time_s": first_forward,
        "minimum_forward_holding_temp_c": round(
            min(float(row["forward_temp_c"]) for row in forward_rows), 4
        )
        if forward_rows
        else None,
        "minimum_forward_actual_residence_time_s": round(
            min(float(row["forward_residence_time_s"]) for row in forward_rows), 4
        )
        if forward_rows
        else None,
        "minimum_forward_fastest_residence_time_s": round(
            min(
                float(row["forward_fastest_residence_time_s"])
                for row in forward_rows
            ),
            4,
        )
        if forward_rows
        else None,
        "minimum_forward_relative_lethality": round(
            min(float(row["forward_relative_lethality"]) for row in forward_rows), 6
        )
        if forward_rows
        else None,
        "minimum_forward_differential_pressure_bar": round(
            min(
                float(row["forward_min_process_differential_pressure_bar"])
                for row in forward_rows
            ),
            4,
        )
        if forward_rows
        else None,
        "maximum_forward_product_temp_c": round(
            max(float(row["potential_product_temp_c"]) for row in forward_rows), 4
        )
        if forward_rows
        else None,
        "maximum_fouling_index": round(
            max(float(row["fouling_index"]) for row in records), 6
        ),
        "final_fouling_index": round(final_fouling, 6),
        "final_surface_total_soil_g": round(
            float(records[-1]["surface_total_soil_g"]), 6
        ),
        "final_surface_hygiene_risk_fraction": round(
            float(records[-1]["surface_hygiene_risk_fraction"]), 9
        ),
        "final_cip_protein_soil_g": round(
            float(records[-1]["cip_protein_soil_g"]), 6
        ),
        "final_cip_mineral_soil_g": round(
            float(records[-1]["cip_mineral_soil_g"]), 6
        ),
        "final_cip_total_soil_g": round(
            float(records[-1]["cip_total_soil_g"]), 6
        ),
        "final_cip_residual_chemical_fraction": round(
            float(records[-1]["cip_residual_chemical_fraction"]), 9
        ),
        "cip_cleaning_complete": bool(records[-1]["cip_cleaning_complete"]),
        "fouling_reduction_fraction": round(
            max(0.0, (initial_fouling - final_fouling) / initial_fouling),
            6,
        )
        if initial_fouling > 0.0
        else None,
        "thermal_energy_kwh": round(thermal_energy, 4),
        "pump_energy_kwh": round(pump_energy, 4),
        "total_energy_kwh": round(thermal_energy + pump_energy, 4),
        "specific_energy_kwh_per_1000l_forward": round(
            (thermal_energy + pump_energy) / total_forward_l * 1000.0,
            4,
        )
        if total_forward_l
        else None,
        "maximum_abs_mass_balance_error_l": max(
            abs(float(row["mass_balance_error_l"])) for row in records
        ),
        "maximum_holding_inventory_error_l": round(
            max(
                abs(float(row["holding_inventory_l"]) - config.holding_tube_volume_l)
                for row in records
            ),
            9,
        ),
        "maximum_fdv_line_inventory_error_l": round(
            max(
                abs(float(row["fdv_line_inventory_l"]) - config.fdv_line_volume_l)
                for row in records
            ),
            9,
        ),
        "maximum_post_fdv_inventory_error_l": round(
            max(
                abs(
                    float(row["post_fdv_inventory_l"])
                    - config.post_fdv_inventory_l
                )
                for row in records
            ),
            9,
        ),
        "maximum_abs_post_fdv_balance_error_l": max(
            abs(float(row["post_fdv_balance_error_l"])) for row in records
        ),
        "maximum_abs_balance_tank_volume_error_l": max(
            abs(float(row["balance_tank_volume_error_l"])) for row in records
        ),
        "maximum_abs_balance_tank_temperature_moment_error_l_c": max(
            abs(float(row["balance_tank_temperature_moment_error_l_c"]))
            for row in records
        ),
        "maximum_balance_tank_pass_count": max(
            float(row["balance_tank_mean_pass_count"]) for row in records
        ),
        "maximum_balance_tank_risk_fraction": max(
            float(row["balance_tank_risk_fraction"]) for row in records
        ),
        "maximum_abs_shadow_hx_energy_balance_error_kj": max(
            abs(float(row[field]))
            for row in records
            for field in (
                "shadow_regenerator_energy_balance_error_kj",
                "shadow_heater_energy_balance_error_kj",
                "shadow_cooler_energy_balance_error_kj",
            )
        ),
        "observable_alarm_row_count": sum(
            int(row["observable_alarm_count"]) > 0 for row in records
        ),
        "diagnostic_alarm_row_count": sum(
            int(row["diagnostic_alarm_count"]) > 0 for row in records
        ),
        "state_durations_s": dict(sorted(state_durations.items())),
        "alarm_durations_s": dict(sorted(alarm_durations.items())),
        "alarm_activation_count": sum(
            event["event_type"] == "ALARM" and event["state"] == "ACTIVE"
            for event in events
        ),
        "assurance_note": (
            "All safety, lethality and contamination fields are simulation diagnostics only; "
            "they are not organism-specific validation or regulatory certification."
        ),
    }
