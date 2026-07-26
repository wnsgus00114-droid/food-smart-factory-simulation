#!/usr/bin/env python3
"""Reusable, dependency-free process components for the HTST v2 surrogate.

These components are deliberately product- and regulation-agnostic.  They
provide auditable volume/attribute mixing, a conservative two-fluid thermal
network, and an empirical two-soil CIP model.  Parameters remain engineering
assumptions until identified against a particular installation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


_EPS = 1e-12


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _nonnegative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def _fraction(name: str, value: float) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class LiquidStream:
    """A finite liquid amount carrying volume-weighted scalar attributes."""

    volume_l: float
    temperature_c: float
    mean_pass_count: float = 0.0
    risk_fraction: float = 0.0
    chemical_fraction: float = 0.0
    product_fraction: float = 1.0

    def __post_init__(self) -> None:
        _nonnegative("volume_l", self.volume_l)
        _finite("temperature_c", self.temperature_c)
        _nonnegative("mean_pass_count", self.mean_pass_count)
        _fraction("risk_fraction", self.risk_fraction)
        _fraction("chemical_fraction", self.chemical_fraction)
        _fraction("product_fraction", self.product_fraction)


@dataclass(frozen=True)
class BalanceTankConfig:
    capacity_l: float = 1_200.0
    initial_volume_l: float = 600.0
    minimum_operating_volume_l: float = 50.0
    initial_temperature_c: float = 4.0
    ambient_temperature_c: float = 20.0
    heat_loss_tau_s: float = 7_200.0
    density_kg_l: float = 1.03
    heat_capacity_kj_kg_k: float = 3.90

    def __post_init__(self) -> None:
        _positive("capacity_l", self.capacity_l)
        _nonnegative("initial_volume_l", self.initial_volume_l)
        _nonnegative("minimum_operating_volume_l", self.minimum_operating_volume_l)
        _finite("initial_temperature_c", self.initial_temperature_c)
        _finite("ambient_temperature_c", self.ambient_temperature_c)
        _positive("heat_loss_tau_s", self.heat_loss_tau_s)
        _positive("density_kg_l", self.density_kg_l)
        _positive("heat_capacity_kj_kg_k", self.heat_capacity_kj_kg_k)
        if self.initial_volume_l > self.capacity_l:
            raise ValueError("initial_volume_l cannot exceed capacity_l")
        if self.minimum_operating_volume_l > self.capacity_l:
            raise ValueError("minimum_operating_volume_l cannot exceed capacity_l")
        if self.initial_volume_l < self.minimum_operating_volume_l:
            raise ValueError(
                "initial_volume_l cannot be below minimum_operating_volume_l"
            )


@dataclass(frozen=True)
class BalanceTankStep:
    outlet: LiquidStream
    volume_l: float
    temperature_c: float
    mean_pass_count: float
    risk_fraction: float
    chemical_fraction: float
    product_fraction: float
    ambient_heat_kj: float
    volume_balance_error_l: float
    temperature_moment_error_l_c: float
    pass_moment_error_l: float
    risk_volume_error_l: float
    chemical_volume_error_l: float
    product_volume_error_l: float


class BalanceTank:
    """Perfectly mixed constant-level tank with explicit return-stream history."""

    def __init__(self, config: BalanceTankConfig | None = None) -> None:
        self.config = config or BalanceTankConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.volume_l = cfg.initial_volume_l
        self.temperature_moment_l_c = cfg.initial_volume_l * cfg.initial_temperature_c
        self.pass_moment_l = 0.0
        self.risk_volume_l = 0.0
        self.chemical_volume_l = 0.0
        self.product_volume_l = cfg.initial_volume_l

    @property
    def temperature_c(self) -> float:
        return (
            self.temperature_moment_l_c / self.volume_l
            if self.volume_l > _EPS
            else self.config.ambient_temperature_c
        )

    @property
    def mean_pass_count(self) -> float:
        return self.pass_moment_l / self.volume_l if self.volume_l > _EPS else 0.0

    @property
    def risk_fraction(self) -> float:
        return self.risk_volume_l / self.volume_l if self.volume_l > _EPS else 0.0

    @property
    def chemical_fraction(self) -> float:
        return self.chemical_volume_l / self.volume_l if self.volume_l > _EPS else 0.0

    @property
    def product_fraction(self) -> float:
        return self.product_volume_l / self.volume_l if self.volume_l > _EPS else 0.0

    def export_state(self) -> dict[str, float]:
        """Return the complete, JSON-safe dynamic tank state."""
        return {
            "volume_l": self.volume_l,
            "temperature_moment_l_c": self.temperature_moment_l_c,
            "pass_moment_l": self.pass_moment_l,
            "risk_volume_l": self.risk_volume_l,
            "chemical_volume_l": self.chemical_volume_l,
            "product_volume_l": self.product_volume_l,
        }

    def import_state(self, state: dict[str, object]) -> None:
        """Validate and atomically restore a state produced by ``export_state``."""
        expected = {
            "volume_l",
            "temperature_moment_l_c",
            "pass_moment_l",
            "risk_volume_l",
            "chemical_volume_l",
            "product_volume_l",
        }
        if set(state) != expected:
            raise ValueError("invalid balance-tank state fields")
        values = {name: _finite(name, state[name]) for name in expected}
        volume = values["volume_l"]
        if not self.config.minimum_operating_volume_l - _EPS <= volume <= self.config.capacity_l + _EPS:
            raise ValueError("balance-tank state volume is outside configured limits")
        for name in (
            "pass_moment_l",
            "risk_volume_l",
            "chemical_volume_l",
            "product_volume_l",
        ):
            if values[name] < -_EPS:
                raise ValueError(f"balance-tank state {name} must be non-negative")
        if values["risk_volume_l"] > volume + _EPS:
            raise ValueError("balance-tank risk volume exceeds total volume")
        if values["chemical_volume_l"] > volume + _EPS:
            raise ValueError("balance-tank chemical volume exceeds total volume")
        if values["product_volume_l"] > volume + _EPS:
            raise ValueError("balance-tank product volume exceeds total volume")
        self.volume_l = volume
        self.temperature_moment_l_c = values["temperature_moment_l_c"]
        self.pass_moment_l = values["pass_moment_l"]
        self.risk_volume_l = values["risk_volume_l"]
        self.chemical_volume_l = values["chemical_volume_l"]
        self.product_volume_l = values["product_volume_l"]

    def makeup_for_target(self, target_volume_l: float, return_volume_l: float, outflow_l: float) -> float:
        """Return raw make-up needed to finish a step at ``target_volume_l``."""
        target = _nonnegative("target_volume_l", target_volume_l)
        returned = _nonnegative("return_volume_l", return_volume_l)
        outflow = _nonnegative("outflow_l", outflow_l)
        if target > self.config.capacity_l:
            raise ValueError("target_volume_l exceeds tank capacity")
        return max(0.0, target + outflow - self.volume_l - returned)

    def step(
        self,
        dt_s: float,
        raw_in: LiquidStream,
        returned: LiquidStream,
        requested_out_l: float,
    ) -> BalanceTankStep:
        dt = _positive("dt_s", dt_s)
        requested = _nonnegative("requested_out_l", requested_out_l)
        cfg = self.config

        before = (
            self.volume_l,
            self.temperature_moment_l_c,
            self.pass_moment_l,
            self.risk_volume_l,
            self.chemical_volume_l,
            self.product_volume_l,
        )
        incoming_volume = raw_in.volume_l + returned.volume_l
        mixed_volume = self.volume_l + incoming_volume
        if mixed_volume > cfg.capacity_l + 1e-9:
            raise ValueError(
                f"balance tank overflow: {mixed_volume:.9g} L > {cfg.capacity_l:.9g} L"
            )
        if requested > mixed_volume + 1e-9:
            raise ValueError("requested tank outflow exceeds available inventory")
        if mixed_volume - requested < cfg.minimum_operating_volume_l - 1e-9:
            raise ValueError("requested tank outflow violates minimum operating volume")

        self.volume_l = mixed_volume
        self.temperature_moment_l_c += (
            raw_in.volume_l * raw_in.temperature_c
            + returned.volume_l * returned.temperature_c
        )
        self.pass_moment_l += (
            raw_in.volume_l * raw_in.mean_pass_count
            + returned.volume_l * returned.mean_pass_count
        )
        self.risk_volume_l += (
            raw_in.volume_l * raw_in.risk_fraction
            + returned.volume_l * returned.risk_fraction
        )
        self.chemical_volume_l += (
            raw_in.volume_l * raw_in.chemical_fraction
            + returned.volume_l * returned.chemical_fraction
        )
        self.product_volume_l += (
            raw_in.volume_l * raw_in.product_fraction
            + returned.volume_l * returned.product_fraction
        )

        mixed = LiquidStream(
            requested,
            self.temperature_c,
            self.mean_pass_count,
            self.risk_fraction,
            self.chemical_fraction,
            self.product_fraction,
        )
        self.volume_l -= requested
        self.temperature_moment_l_c -= requested * mixed.temperature_c
        self.pass_moment_l -= requested * mixed.mean_pass_count
        self.risk_volume_l -= requested * mixed.risk_fraction
        self.chemical_volume_l -= requested * mixed.chemical_fraction
        self.product_volume_l -= requested * mixed.product_fraction

        # Heat loss acts on the remaining well-mixed inventory.  It is tracked
        # as an external energy term and therefore does not pollute mixing balance.
        ambient_heat_kj = 0.0
        if self.volume_l > _EPS:
            old_temp = self.temperature_c
            factor = 1.0 - math.exp(-dt / cfg.heat_loss_tau_s)
            new_temp = old_temp + factor * (cfg.ambient_temperature_c - old_temp)
            delta_moment = self.volume_l * (new_temp - old_temp)
            self.temperature_moment_l_c += delta_moment
            ambient_heat_kj = (
                delta_moment * cfg.density_kg_l * cfg.heat_capacity_kj_kg_k
            )

        expected_volume = before[0] + incoming_volume - requested
        expected_temp_moment = (
            before[1]
            + raw_in.volume_l * raw_in.temperature_c
            + returned.volume_l * returned.temperature_c
            - requested * mixed.temperature_c
            + ambient_heat_kj / (cfg.density_kg_l * cfg.heat_capacity_kj_kg_k)
        )
        expected_pass = (
            before[2]
            + raw_in.volume_l * raw_in.mean_pass_count
            + returned.volume_l * returned.mean_pass_count
            - requested * mixed.mean_pass_count
        )
        expected_risk = (
            before[3]
            + raw_in.volume_l * raw_in.risk_fraction
            + returned.volume_l * returned.risk_fraction
            - requested * mixed.risk_fraction
        )
        expected_chemical = (
            before[4]
            + raw_in.volume_l * raw_in.chemical_fraction
            + returned.volume_l * returned.chemical_fraction
            - requested * mixed.chemical_fraction
        )
        expected_product = (
            before[5]
            + raw_in.volume_l * raw_in.product_fraction
            + returned.volume_l * returned.product_fraction
            - requested * mixed.product_fraction
        )

        return BalanceTankStep(
            outlet=mixed,
            volume_l=self.volume_l,
            temperature_c=self.temperature_c,
            mean_pass_count=self.mean_pass_count,
            risk_fraction=self.risk_fraction,
            chemical_fraction=self.chemical_fraction,
            product_fraction=self.product_fraction,
            ambient_heat_kj=ambient_heat_kj,
            volume_balance_error_l=self.volume_l - expected_volume,
            temperature_moment_error_l_c=self.temperature_moment_l_c - expected_temp_moment,
            pass_moment_error_l=self.pass_moment_l - expected_pass,
            risk_volume_error_l=self.risk_volume_l - expected_risk,
            chemical_volume_error_l=self.chemical_volume_l - expected_chemical,
            product_volume_error_l=self.product_volume_l - expected_product,
        )


@dataclass(frozen=True)
class HeatExchangerConfig:
    clean_ua_kw_k: float = 40.0
    product_holdup_l: float = 25.0
    utility_holdup_l: float = 25.0
    wall_heat_capacity_kj_k: float = 60.0
    product_density_kg_l: float = 1.03
    product_heat_capacity_kj_kg_k: float = 3.90
    utility_density_kg_l: float = 1.0
    utility_heat_capacity_kj_kg_k: float = 4.18
    ambient_ua_kw_k: float = 0.02
    ambient_temperature_c: float = 20.0
    fouling_resistance_ratio_at_one: float = 4.0
    internal_max_dt_s: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "clean_ua_kw_k",
            "product_holdup_l",
            "utility_holdup_l",
            "wall_heat_capacity_kj_k",
            "product_density_kg_l",
            "product_heat_capacity_kj_kg_k",
            "utility_density_kg_l",
            "utility_heat_capacity_kj_kg_k",
            "internal_max_dt_s",
        ):
            _positive(name, getattr(self, name))
        _nonnegative("ambient_ua_kw_k", self.ambient_ua_kw_k)
        _finite("ambient_temperature_c", self.ambient_temperature_c)
        _nonnegative(
            "fouling_resistance_ratio_at_one", self.fouling_resistance_ratio_at_one
        )


@dataclass(frozen=True)
class HeatExchangerStep:
    product_outlet_c: float
    utility_outlet_c: float
    wall_temperature_c: float
    effective_ua_kw_k: float
    product_heat_kj: float
    utility_heat_kj: float
    ambient_heat_kj: float
    stored_energy_change_kj: float
    energy_balance_error_kj: float


class DynamicHeatExchanger:
    """Conservative two-fluid/wall lumped model with internal sub-stepping."""

    def __init__(
        self,
        config: HeatExchangerConfig | None = None,
        *,
        initial_product_c: float = 4.0,
        initial_utility_c: float = 80.0,
        initial_wall_c: float | None = None,
    ) -> None:
        self.config = config or HeatExchangerConfig()
        self.initial_product_c = _finite("initial_product_c", initial_product_c)
        self.initial_utility_c = _finite("initial_utility_c", initial_utility_c)
        self.initial_wall_c = _finite(
            "initial_wall_c",
            0.5 * (initial_product_c + initial_utility_c)
            if initial_wall_c is None
            else initial_wall_c,
        )
        self.reset()

    def reset(self) -> None:
        self.product_bulk_c = self.initial_product_c
        self.utility_bulk_c = self.initial_utility_c
        self.wall_temperature_c = self.initial_wall_c

    def export_state(self) -> dict[str, float]:
        """Return all thermal storage states required for exact continuation."""
        return {
            "product_bulk_c": self.product_bulk_c,
            "utility_bulk_c": self.utility_bulk_c,
            "wall_temperature_c": self.wall_temperature_c,
        }

    def import_state(self, state: dict[str, object]) -> None:
        expected = {
            "product_bulk_c",
            "utility_bulk_c",
            "wall_temperature_c",
        }
        if set(state) != expected:
            raise ValueError("invalid heat-exchanger state fields")
        values = {name: _finite(name, state[name]) for name in expected}
        self.product_bulk_c = values["product_bulk_c"]
        self.utility_bulk_c = values["utility_bulk_c"]
        self.wall_temperature_c = values["wall_temperature_c"]

    @property
    def _product_capacity_kj_k(self) -> float:
        cfg = self.config
        return cfg.product_holdup_l * cfg.product_density_kg_l * cfg.product_heat_capacity_kj_kg_k

    @property
    def _utility_capacity_kj_k(self) -> float:
        cfg = self.config
        return cfg.utility_holdup_l * cfg.utility_density_kg_l * cfg.utility_heat_capacity_kj_kg_k

    def _stored_energy(self) -> float:
        return (
            self._product_capacity_kj_k * self.product_bulk_c
            + self._utility_capacity_kj_k * self.utility_bulk_c
            + self.config.wall_heat_capacity_kj_k * self.wall_temperature_c
        )

    @staticmethod
    def _exchange_pair(
        first_temp: float,
        first_capacity: float,
        second_temp: float,
        second_capacity: float,
        conductance_kw_k: float,
        dt_s: float,
    ) -> tuple[float, float]:
        if conductance_kw_k <= 0.0 or dt_s <= 0.0:
            return first_temp, second_temp
        mean = (
            first_capacity * first_temp + second_capacity * second_temp
        ) / (first_capacity + second_capacity)
        decay = math.exp(
            -conductance_kw_k
            * (1.0 / first_capacity + 1.0 / second_capacity)
            * dt_s
        )
        difference = (first_temp - second_temp) * decay
        new_first = mean + second_capacity / (first_capacity + second_capacity) * difference
        new_second = mean - first_capacity / (first_capacity + second_capacity) * difference
        return new_first, new_second

    def step(
        self,
        dt_s: float,
        *,
        product_inlet_c: float,
        product_flow_l_s: float,
        utility_inlet_c: float,
        utility_flow_l_s: float,
        fouling_index: float = 0.0,
    ) -> HeatExchangerStep:
        dt = _positive("dt_s", dt_s)
        product_inlet = _finite("product_inlet_c", product_inlet_c)
        utility_inlet = _finite("utility_inlet_c", utility_inlet_c)
        product_flow = _nonnegative("product_flow_l_s", product_flow_l_s)
        utility_flow = _nonnegative("utility_flow_l_s", utility_flow_l_s)
        fouling = _fraction("fouling_index", fouling_index)
        cfg = self.config
        effective_ua = cfg.clean_ua_kw_k / (
            1.0 + cfg.fouling_resistance_ratio_at_one * fouling
        )
        # Each film has twice the overall conductance; two equal film
        # resistances in series then recover effective_ua.
        film_ua = 2.0 * effective_ua
        steps = max(1, math.ceil(dt / cfg.internal_max_dt_s))
        sub_dt = dt / steps
        initial_energy = self._stored_energy()
        product_advective_kj = 0.0
        utility_advective_kj = 0.0
        ambient_heat_kj = 0.0
        product_capacity = self._product_capacity_kj_k
        utility_capacity = self._utility_capacity_kj_k
        wall_capacity = cfg.wall_heat_capacity_kj_k

        for _ in range(steps):
            if product_flow > 0.0:
                old = self.product_bulk_c
                mix = 1.0 - math.exp(-product_flow * sub_dt / cfg.product_holdup_l)
                self.product_bulk_c += mix * (product_inlet - self.product_bulk_c)
                product_advective_kj += product_capacity * (self.product_bulk_c - old)
            if utility_flow > 0.0:
                old = self.utility_bulk_c
                mix = 1.0 - math.exp(-utility_flow * sub_dt / cfg.utility_holdup_l)
                self.utility_bulk_c += mix * (utility_inlet - self.utility_bulk_c)
                utility_advective_kj += utility_capacity * (self.utility_bulk_c - old)

            self.product_bulk_c, self.wall_temperature_c = self._exchange_pair(
                self.product_bulk_c,
                product_capacity,
                self.wall_temperature_c,
                wall_capacity,
                film_ua,
                sub_dt,
            )
            self.utility_bulk_c, self.wall_temperature_c = self._exchange_pair(
                self.utility_bulk_c,
                utility_capacity,
                self.wall_temperature_c,
                wall_capacity,
                film_ua,
                sub_dt,
            )
            if cfg.ambient_ua_kw_k > 0.0:
                old_wall = self.wall_temperature_c
                factor = 1.0 - math.exp(
                    -cfg.ambient_ua_kw_k * sub_dt / wall_capacity
                )
                self.wall_temperature_c += factor * (
                    cfg.ambient_temperature_c - self.wall_temperature_c
                )
                ambient_heat_kj += wall_capacity * (
                    self.wall_temperature_c - old_wall
                )

        stored_change = self._stored_energy() - initial_energy
        balance_error = stored_change - (
            product_advective_kj + utility_advective_kj + ambient_heat_kj
        )
        values = (
            self.product_bulk_c,
            self.utility_bulk_c,
            self.wall_temperature_c,
            effective_ua,
            product_advective_kj,
            utility_advective_kj,
            ambient_heat_kj,
            stored_change,
            balance_error,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("heat exchanger produced a non-finite state")
        return HeatExchangerStep(
            product_outlet_c=self.product_bulk_c,
            utility_outlet_c=self.utility_bulk_c,
            wall_temperature_c=self.wall_temperature_c,
            effective_ua_kw_k=effective_ua,
            product_heat_kj=product_advective_kj,
            utility_heat_kj=utility_advective_kj,
            ambient_heat_kj=ambient_heat_kj,
            stored_energy_change_kj=stored_change,
            energy_balance_error_kj=balance_error,
        )


@dataclass(frozen=True)
class CIPSoilConfig:
    initial_protein_soil_g: float = 650.0
    initial_mineral_soil_g: float = 350.0
    protein_reference_rate_s: float = 0.006
    mineral_reference_rate_s: float = 0.003
    reference_velocity_m_s: float = 1.5
    line_hold_up_l: float = 100.0
    clean_soil_threshold_g: float = 10.0
    residual_chemical_threshold_fraction: float = 0.002
    ambient_temperature_c: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "initial_protein_soil_g",
            "initial_mineral_soil_g",
            "protein_reference_rate_s",
            "mineral_reference_rate_s",
            "reference_velocity_m_s",
            "line_hold_up_l",
            "clean_soil_threshold_g",
        ):
            _positive(name, getattr(self, name))
        _fraction(
            "residual_chemical_threshold_fraction",
            self.residual_chemical_threshold_fraction,
        )
        _finite("ambient_temperature_c", self.ambient_temperature_c)
        if self.clean_soil_threshold_g > (
            self.initial_protein_soil_g + self.initial_mineral_soil_g
        ):
            raise ValueError(
                "clean_soil_threshold_g cannot exceed initial total soil"
            )


@dataclass(frozen=True)
class CIPStep:
    protein_soil_g: float
    mineral_soil_g: float
    total_soil_g: float
    removed_protein_g: float
    removed_mineral_g: float
    residual_chemical_fraction: float
    conductivity_proxy_ms_cm: float
    ph_proxy: float
    cleaning_complete: bool


class CIPSoilModel:
    """Empirical two-soil cleaning model with rinse carry-over tracking."""

    def __init__(self, config: CIPSoilConfig | None = None) -> None:
        self.config = config or CIPSoilConfig()
        self.reset()

    def reset(self) -> None:
        self.protein_soil_g = self.config.initial_protein_soil_g
        self.mineral_soil_g = self.config.initial_mineral_soil_g
        self.residual_alkali_fraction = 0.0
        self.residual_acid_fraction = 0.0

    def export_state(self) -> dict[str, float]:
        """Return soil and line-chemical states for campaign checkpoints."""
        return {
            "protein_soil_g": self.protein_soil_g,
            "mineral_soil_g": self.mineral_soil_g,
            "residual_alkali_fraction": self.residual_alkali_fraction,
            "residual_acid_fraction": self.residual_acid_fraction,
        }

    def import_state(self, state: dict[str, object]) -> None:
        expected = {
            "protein_soil_g",
            "mineral_soil_g",
            "residual_alkali_fraction",
            "residual_acid_fraction",
        }
        if set(state) != expected:
            raise ValueError("invalid CIP-soil state fields")
        protein = _nonnegative("protein_soil_g", state["protein_soil_g"])
        mineral = _nonnegative("mineral_soil_g", state["mineral_soil_g"])
        alkali = _fraction(
            "residual_alkali_fraction", state["residual_alkali_fraction"]
        )
        acid = _fraction(
            "residual_acid_fraction", state["residual_acid_fraction"]
        )
        if alkali + acid > 1.0 + _EPS:
            raise ValueError("combined CIP residual chemical exceeds 1")
        self.protein_soil_g = protein
        self.mineral_soil_g = mineral
        self.residual_alkali_fraction = alkali
        self.residual_acid_fraction = acid

    def set_soil(self, protein_soil_g: float, mineral_soil_g: float) -> None:
        """Atomically replace the modeled surface soil without changing chemistry."""
        protein = _nonnegative("protein_soil_g", protein_soil_g)
        mineral = _nonnegative("mineral_soil_g", mineral_soil_g)
        self.protein_soil_g = protein
        self.mineral_soil_g = mineral

    def deposit_soil(self, protein_soil_g: float, mineral_soil_g: float) -> None:
        """Add production soil while preserving the two-soil composition."""
        protein = _nonnegative("protein_soil_g", protein_soil_g)
        mineral = _nonnegative("mineral_soil_g", mineral_soil_g)
        self.protein_soil_g += protein
        self.mineral_soil_g += mineral

    @property
    def total_soil_g(self) -> float:
        return self.protein_soil_g + self.mineral_soil_g

    @property
    def residual_chemical_fraction(self) -> float:
        return min(1.0, self.residual_alkali_fraction + self.residual_acid_fraction)

    def step(
        self,
        dt_s: float,
        *,
        temperature_c: float,
        velocity_m_s: float,
        alkali_pct: float = 0.0,
        acid_pct: float = 0.0,
        rinse_flow_l_s: float = 0.0,
    ) -> CIPStep:
        dt = _positive("dt_s", dt_s)
        temperature = _finite("temperature_c", temperature_c)
        velocity = _nonnegative("velocity_m_s", velocity_m_s)
        alkali = _nonnegative("alkali_pct", alkali_pct)
        acid = _nonnegative("acid_pct", acid_pct)
        rinse_flow = _nonnegative("rinse_flow_l_s", rinse_flow_l_s)
        if alkali > 10.0 or acid > 10.0:
            raise ValueError("chemical concentration proxy must be <= 10 percent")
        if alkali > 0.0 and acid > 0.0:
            raise ValueError("alkali and acid cannot be active simultaneously")
        cfg = self.config

        hydrodynamic = min(2.0, (velocity / cfg.reference_velocity_m_s) ** 0.8) if velocity else 0.0
        protein_temp = min(2.0, max(0.0, (temperature - 35.0) / 40.0))
        mineral_temp = min(2.0, max(0.0, (temperature - 30.0) / 40.0))
        protein_rate = (
            cfg.protein_reference_rate_s
            * hydrodynamic
            * protein_temp
            * min(2.0, alkali / 1.0)
        )
        mineral_rate = (
            cfg.mineral_reference_rate_s
            * hydrodynamic
            * mineral_temp
            * min(2.0, acid / 0.8)
        )
        old_protein = self.protein_soil_g
        old_mineral = self.mineral_soil_g
        self.protein_soil_g *= math.exp(-protein_rate * dt)
        self.mineral_soil_g *= math.exp(-mineral_rate * dt)

        # Perfect-mixing displacement of the chemical hold-up.  The
        # exponential form is stable for large time steps and composes exactly
        # over sub-steps, unlike a clipped linear replacement fraction.
        displacement = 1.0 - math.exp(-rinse_flow * dt / cfg.line_hold_up_l)
        if alkali > 0.0:
            self.residual_alkali_fraction += displacement * (
                min(1.0, alkali / 100.0) - self.residual_alkali_fraction
            )
            self.residual_acid_fraction *= 1.0 - displacement
        elif acid > 0.0:
            self.residual_acid_fraction += displacement * (
                min(1.0, acid / 100.0) - self.residual_acid_fraction
            )
            self.residual_alkali_fraction *= 1.0 - displacement
        elif rinse_flow > 0.0:
            self.residual_alkali_fraction *= 1.0 - displacement
            self.residual_acid_fraction *= 1.0 - displacement

        residual = self.residual_chemical_fraction
        conductivity = 0.2 + 65.0 * self.residual_alkali_fraction + 45.0 * self.residual_acid_fraction
        ph = 7.0 + 5.0 * math.tanh(80.0 * self.residual_alkali_fraction) - 4.0 * math.tanh(
            80.0 * self.residual_acid_fraction
        )
        complete = (
            self.total_soil_g <= cfg.clean_soil_threshold_g
            and residual <= cfg.residual_chemical_threshold_fraction
        )
        return CIPStep(
            protein_soil_g=self.protein_soil_g,
            mineral_soil_g=self.mineral_soil_g,
            total_soil_g=self.total_soil_g,
            removed_protein_g=old_protein - self.protein_soil_g,
            removed_mineral_g=old_mineral - self.mineral_soil_g,
            residual_chemical_fraction=residual,
            conductivity_proxy_ms_cm=conductivity,
            ph_proxy=ph,
            cleaning_complete=complete,
        )
