#!/usr/bin/env python3
"""Focused physics and numerical-contract tests for simulator v2."""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import (  # noqa: E402
    ALARM_COLUMNS,
    DIAGNOSTIC_ALARM_COLUMNS,
    MODEL_VERSION,
    HTSTConfig,
    HTSTSimulator,
    ScenarioModifiers,
    ThermalSlice,
    summarize,
)


ZERO_NOISE = {
    "preheat_sensor_noise_std_c": 0.0,
    "control_sensor_noise_std_c": 0.0,
    "safety_sensor_noise_std_c": 0.0,
    "flow_meter_noise_std_fraction": 0.0,
    "pressure_sensor_noise_std_bar": 0.0,
    "booster_speed_sensor_noise_std_fraction": 0.0,
    "leak_detector_noise_std_fraction": 0.0,
    "product_sensor_noise_std_c": 0.0,
    "fdv_position_sensor_noise_std_fraction": 0.0,
}


def quiet_config(**changes: float | int) -> HTSTConfig:
    values: dict[str, float | int] = dict(ZERO_NOISE)
    values.update(changes)
    return replace(HTSTConfig(), **values)


def numeric_values(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from numeric_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from numeric_values(item)
    elif isinstance(value, (int, float)):
        yield float(value)


class V2NumericTests(unittest.TestCase):
    def test_version_dt_step_and_numeric_bounds(self) -> None:
        self.assertEqual(MODEL_VERSION, "2.2.0")
        config = HTSTConfig()
        invalid = (
            {"dt_s": 0.0005},
            {"maximum_steps": 10, "duration_s": 20.0, "dt_s": 1.0},
            {"maximum_steps": True},
            {"fastest_flow_efficiency": 0.0},
            {"fastest_flow_efficiency": 1.01},
            {"max_exponential_argument": 701.0},
            {"ambient_temp_c": -274.0},
            {"stationary_cooling_tau_s": 0.0},
            {"post_fdv_residence_time_s": 0.0},
            {"dt_s": "0.5"},
            {"duration_s": 1.0e308, "dt_s": 0.001},
            {"balance_tank_capacity_l": 1.0e308},
            {"hx_regenerator_clean_ua_kw_k": 1.0e308},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(config, **changes)

    def test_bounded_exponential_diagnostic_remains_finite(self) -> None:
        config = quiet_config(duration_s=1.0, max_exponential_argument=20.0)
        simulator = HTSTSimulator(config)
        for parcel in simulator.holding_queue:
            parcel.temp_c = 1_000.0
        volume_l = config.nominal_flow_l_s * config.dt_s
        outlet = simulator._move_through_holding_tube(
            volume_l, 1_000.0, 0.0, config.dt_s
        )
        self.assertGreater(outlet.bounded_exponential_count, 0)
        self.assertTrue(math.isfinite(outlet.mean_relative_lethality))
        self.assertLessEqual(
            outlet.mean_relative_lethality,
            math.exp(config.max_exponential_argument),
        )

    def test_every_new_scenario_is_deterministic_and_finite(self) -> None:
        config = quiet_config(
            duration_s=100.0,
            dt_s=0.5,
            fault_start_s=45.0,
            fault_duration_s=20.0,
        )
        scenarios = (
            "start_stop",
            "incomplete_cleaning",
            "sensor_drift",
            "sensor_dropout",
            "slow_valve",
            "valve_leakage",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                first = HTSTSimulator(config).run(scenario)
                second = HTSTSimulator(config).run(scenario)
                self.assertEqual(first, second)
                summary = summarize(first, config)
                self.assertTrue(all(math.isfinite(value) for value in numeric_values(first)))
                self.assertTrue(all(math.isfinite(value) for value in numeric_values(summary)))
                json.dumps(first, allow_nan=False)
                json.dumps(summary, allow_nan=False)

    def test_sensor_dropout_preserves_channelwise_counterfactual_noise(self) -> None:
        config = HTSTConfig(
            duration_s=40.0,
            dt_s=0.5,
            fault_start_s=15.0,
            fault_duration_s=10.0,
            random_seed=424242,
        )
        normal = HTSTSimulator(config).run("normal")
        dropout = HTSTSimulator(config).run("sensor_dropout")
        self.assertEqual(len(normal), len(dropout))
        for reference, faulted in zip(normal, dropout, strict=True):
            self.assertEqual(reference["time_s"], faulted["time_s"])
            if not int(faulted["fault_active"]):
                continue
            reference_noise = {
                "flow": float(reference["measured_flow_l_h"])
                / float(reference["flow_l_h"])
                - 1.0,
                "raw_pressure": float(reference["raw_pressure_sensor_bar"])
                - float(reference["raw_pressure_bar"]),
                "pasteurized_pressure": float(
                    reference["pasteurized_pressure_sensor_bar"]
                )
                - float(reference["pasteurized_pressure_bar"]),
                "product": float(reference["product_temp_sensor_c"])
                - float(reference["potential_product_temp_c"]),
            }
            faulted_noise = {
                "flow": float(faulted["measured_flow_l_h"])
                / float(faulted["flow_l_h"])
                - 1.0,
                "raw_pressure": float(faulted["raw_pressure_sensor_bar"])
                - float(faulted["raw_pressure_bar"]),
                "pasteurized_pressure": float(
                    faulted["pasteurized_pressure_sensor_bar"]
                )
                - float(faulted["pasteurized_pressure_bar"]),
                "product": float(faulted["product_temp_sensor_c"])
                - float(faulted["potential_product_temp_c"]),
            }
            for channel in reference_noise:
                self.assertAlmostEqual(
                    reference_noise[channel], faulted_noise[channel], places=12
                )


class V2TransportTests(unittest.TestCase):
    def test_stationary_inventories_relax_exactly_toward_ambient(self) -> None:
        config = quiet_config(
            duration_s=1.0,
            stationary_cooling_tau_s=4.0,
            ambient_temp_c=20.0,
        )
        simulator = HTSTSimulator(config)
        for parcel in simulator.holding_queue:
            parcel.temp_c = 80.0
        for item in simulator.fdv_line_queue:
            item.temp_c = 70.0
        for item in simulator.post_fdv_queue:
            item.temp_c = 60.0
            item.product_temp_c = 50.0
        simulator.last_hold_temp_c = 80.0
        inventories_before = (
            sum(item.volume_l for item in simulator.holding_queue),
            sum(item.volume_l for item in simulator.fdv_line_queue),
            sum(item.volume_l for item in simulator.post_fdv_queue),
        )

        dt_s = 2.0
        simulator._move_through_holding_tube(0.0, 80.0, 0.0, dt_s)
        simulator._relax_stationary_downstream(dt_s)
        decay = math.exp(-dt_s / config.stationary_cooling_tau_s)
        self.assertAlmostEqual(
            simulator.holding_queue[0].temp_c,
            config.ambient_temp_c + (80.0 - config.ambient_temp_c) * decay,
            places=12,
        )
        self.assertAlmostEqual(
            simulator.fdv_line_queue[0].temp_c,
            config.ambient_temp_c + (70.0 - config.ambient_temp_c) * decay,
            places=12,
        )
        self.assertAlmostEqual(
            simulator.post_fdv_queue[0].product_temp_c,
            config.ambient_temp_c + (50.0 - config.ambient_temp_c) * decay,
            places=12,
        )
        inventories_after = (
            sum(item.volume_l for item in simulator.holding_queue),
            sum(item.volume_l for item in simulator.fdv_line_queue),
            sum(item.volume_l for item in simulator.post_fdv_queue),
        )
        self.assertEqual(inventories_after, inventories_before)

    def test_post_fdv_fifo_preserves_partial_slice_metadata(self) -> None:
        config = quiet_config(
            duration_s=1.0,
            nominal_flow_l_h=3_600.0,
            post_fdv_residence_time_s=2.5,
        )
        simulator = HTSTSimulator(config)

        def move(volume_l: float, tag: float, fault: bool):
            return simulator._move_post_fdv(
                (
                    ThermalSlice(
                        volume_l,
                        tag,
                        10.0,
                        1.0,
                        True,
                        fastest_residence_time_s=8.5,
                        downstream_fault_active=fault,
                    ),
                ),
                effective_regeneration=0.0,
                cooler_factor=1.0,
                inlet_temp_c=4.0,
                downstream_fault_active=fault,
                process_differential_pressure_bar=1.0,
                pressure_safe=True,
                contamination_risk=False,
            )

        move(0.75, 101.0, True)
        move(1.25, 102.0, False)
        third = move(1.0, 103.0, False)
        self.assertEqual([round(item.volume_l, 12) for item in third], [0.5, 0.5])
        self.assertEqual([item.temp_c for item in third], [74.0, 101.0])
        self.assertTrue(third[1].downstream_fault_active)

        fourth = move(0.5, 104.0, False)
        self.assertEqual([round(item.volume_l, 12) for item in fourth], [0.25, 0.25])
        self.assertEqual([item.temp_c for item in fourth], [101.0, 102.0])
        self.assertAlmostEqual(
            sum(item.volume_l for item in simulator.post_fdv_queue),
            config.post_fdv_inventory_l,
            places=12,
        )

    def test_post_fdv_entry_uses_current_pressure_and_leak_state(self) -> None:
        config = quiet_config(
            duration_s=1.0,
            nominal_flow_l_h=3_600.0,
            post_fdv_residence_time_s=2.5,
        )
        simulator = HTSTSimulator(config)
        stale_sensor_slice = ThermalSlice(
            1.0,
            74.0,
            15.0,
            1.0,
            True,
            pressure_safe=True,
            contamination_risk=False,
            process_differential_pressure_bar=1.0,
        )

        simulator._move_post_fdv(
            (stale_sensor_slice,),
            effective_regeneration=0.9,
            cooler_factor=1.0,
            inlet_temp_c=4.0,
            downstream_fault_active=True,
            process_differential_pressure_bar=-0.2,
            pressure_safe=False,
            contamination_risk=True,
        )

        entered = simulator.post_fdv_queue[-1]
        self.assertAlmostEqual(entered.process_differential_pressure_bar, -0.2)
        self.assertFalse(entered.pressure_safe)
        self.assertTrue(entered.contamination_risk)

    def test_downstream_fault_reaches_product_boundary_after_fifo_delay(self) -> None:
        config = quiet_config(
            duration_s=90.0,
            dt_s=0.5,
            fault_start_s=60.0,
            fault_duration_s=15.0,
            post_fdv_residence_time_s=5.0,
        )
        records = HTSTSimulator(config).run("cooling_utility_loss")
        first_fault = next(row for row in records if int(row["fault_active"]))
        first_product_fault = next(
            row
            for row in records
            if float(row["product_downstream_fault_fraction"]) > 0.0
        )
        self.assertEqual(float(first_fault["time_start_s"]), config.fault_start_s)
        self.assertAlmostEqual(
            float(first_product_fault["time_start_s"]),
            config.fault_start_s + config.post_fdv_residence_time_s,
            places=9,
        )
        before_delay = [
            row
            for row in records
            if config.fault_start_s
            <= float(row["time_start_s"])
            < config.fault_start_s + config.post_fdv_residence_time_s
        ]
        self.assertTrue(
            all(float(row["product_downstream_fault_fraction"]) == 0.0 for row in before_delay)
        )


class V2ValveAndBalanceTests(unittest.TestCase):
    def test_recovery_confirmation_keeps_closing_partly_open_valve(self) -> None:
        config = quiet_config(
            duration_s=1.0,
            dt_s=0.1,
            nominal_flow_l_h=3_600.0,
            fdv_travel_time_s=1.0,
            sensor_to_fdv_delay_s=2.0,
            forward_confirmation_s=1.0,
        )
        simulator = HTSTSimulator(config)
        simulator._fdv_position = 0.5
        simulator.forward_permissive_confirmed = False

        start_fraction, end_fraction = simulator._update_fdv(
            True,
            ScenarioModifiers(),
            step_dt_s=0.1,
            observed_safe_volume_l=0.1,
        )

        self.assertAlmostEqual(simulator._fdv_position, 0.4)
        self.assertAlmostEqual(start_fraction, 0.0)
        self.assertAlmostEqual(end_fraction, 0.45)
        self.assertFalse(simulator.forward_permissive_confirmed)

    def test_main_loop_recycles_diverted_liquid_through_balance_tank(self) -> None:
        config = quiet_config(duration_s=45.0, dt_s=0.5)
        records = HTSTSimulator(config).run("normal")
        summary = summarize(records, config)

        self.assertTrue(
            any(float(row["return_to_balance_tank_l"]) > 0.0 for row in records)
        )
        self.assertGreater(
            max(float(row["balance_tank_mean_pass_count"]) for row in records),
            0.0,
        )
        self.assertGreater(
            max(float(row["balance_tank_temp_c"]) for row in records),
            config.raw_milk_temp_c,
        )
        self.assertAlmostEqual(
            float(summary["external_volume_balance_error_l"]), 0.0, places=9
        )
        self.assertAlmostEqual(
            float(summary["maximum_abs_balance_tank_volume_error_l"]),
            0.0,
            places=9,
        )

    def test_detailed_cip_soils_distinguish_incomplete_cleaning(self) -> None:
        config = quiet_config(
            duration_s=550.0,
            dt_s=0.5,
            fault_start_s=300.0,
            fault_duration_s=120.0,
        )
        complete = HTSTSimulator(config).run("cip_cycle")
        incomplete = HTSTSimulator(config).run("incomplete_cleaning")
        self.assertLess(
            float(complete[-1]["cip_total_soil_g"]),
            float(incomplete[-1]["cip_total_soil_g"]),
        )
        self.assertLess(
            float(complete[-1]["fouling_index"]),
            float(incomplete[-1]["fouling_index"]),
        )
        self.assertEqual(int(complete[-1]["cip_cleaning_complete"]), 1)
        self.assertEqual(int(incomplete[-1]["cip_cleaning_complete"]), 0)
        self.assertTrue(
            any(float(row["cip_removed_protein_g"]) > 0.0 for row in complete)
        )
        self.assertTrue(
            any(float(row["cip_removed_mineral_g"]) > 0.0 for row in complete)
        )
        production_ccp_alarms = (
            "alarm_low_temperature",
            "alarm_low_holding_time",
            "alarm_high_flow",
            "alarm_low_pressure_differential",
        )
        for records in (complete, incomplete):
            self.assertTrue(
                all(
                    int(row[field]) == 0
                    for row in records
                    for field in production_ccp_alarms
                ),
                "milk-production CCP alarms must be inhibited during CIP",
            )

    def test_shadow_heat_exchanger_network_is_integrated_and_conservative(self) -> None:
        config = quiet_config(duration_s=90.0, dt_s=0.5)
        records = HTSTSimulator(config).run("normal")
        summary = summarize(records, config)
        for row in records:
            for field in (
                "shadow_regenerator_energy_balance_error_kj",
                "shadow_heater_energy_balance_error_kj",
                "shadow_cooler_energy_balance_error_kj",
            ):
                self.assertLess(abs(float(row[field])), 1e-8)
        self.assertLess(
            float(summary["maximum_abs_shadow_hx_energy_balance_error_kj"]),
            1e-8,
        )
        self.assertTrue(
            all(
                math.isfinite(float(row[field]))
                for row in records
                for field in (
                    "shadow_regenerator_cold_out_c",
                    "shadow_heater_product_out_c",
                    "shadow_cooler_product_out_c",
                )
            )
        )

    def test_valve_travel_is_fractional_and_integrates_triangular_closure(self) -> None:
        config = quiet_config(
            duration_s=1.0,
            dt_s=0.2,
            divert_actuation_delay_s=0.0,
            fdv_travel_time_s=0.8,
            sensor_to_fdv_delay_s=0.0,
            forward_confirmation_s=0.0,
        )
        simulator = HTSTSimulator(config)
        simulator.fdv_forward = True
        area_s = 0.0
        for _ in range(4):
            start, end = simulator._update_fdv(
                False, ScenarioModifiers(), config.dt_s, 0.0
            )
            area_s += (end - start) * config.dt_s
        self.assertAlmostEqual(area_s, 0.4, places=12)
        self.assertAlmostEqual(simulator._fdv_position, 0.0, places=12)

    def test_leakage_floor_does_not_bypass_forward_confirmation(self) -> None:
        config = quiet_config(duration_s=1.0, dt_s=0.5)
        simulator = HTSTSimulator(config)
        modifiers = ScenarioModifiers(valve_leakage_fraction=0.08)
        volume_l = config.nominal_flow_l_s * config.dt_s
        required_steps = math.ceil(
            (
                config.fdv_line_volume_l
                + config.nominal_flow_l_s * config.forward_confirmation_s
            )
            / volume_l
            - 1e-12
        )
        for _ in range(required_steps - 1):
            simulator._update_fdv(True, modifiers, config.dt_s, volume_l)
            self.assertFalse(simulator.forward_permissive_confirmed)
            self.assertAlmostEqual(simulator._fdv_position, 0.08)
        simulator._update_fdv(True, modifiers, config.dt_s, volume_l)
        self.assertTrue(simulator.forward_permissive_confirmed)

    def test_new_scenarios_conserve_all_three_fixed_inventories(self) -> None:
        config = quiet_config(
            duration_s=120.0,
            dt_s=0.5,
            fault_start_s=55.0,
            fault_duration_s=25.0,
        )
        scenarios = (
            "start_stop",
            "incomplete_cleaning",
            "sensor_drift",
            "sensor_dropout",
            "slow_valve",
            "valve_leakage",
        )
        for scenario in scenarios:
            records = HTSTSimulator(config).run(scenario)
            with self.subTest(scenario=scenario):
                for row in records:
                    self.assertAlmostEqual(
                        float(row["routed_volume_l"]),
                        float(row["forward_l"])
                        + float(row["diverted_l"])
                        + float(row["cip_recirculated_l"]),
                        places=9,
                    )
                    self.assertAlmostEqual(float(row["mass_balance_error_l"]), 0.0, places=9)
                    self.assertAlmostEqual(
                        float(row["post_fdv_balance_error_l"]), 0.0, places=9
                    )
                    self.assertAlmostEqual(
                        float(row["safe_forward_l"])
                        + float(row["unsafe_forward_l"]),
                        float(row["forward_l"]),
                        places=9,
                    )
                    self.assertAlmostEqual(
                        float(row["holding_inventory_l"]),
                        config.holding_tube_volume_l,
                        places=8,
                    )
                    self.assertAlmostEqual(
                        float(row["fdv_line_inventory_l"]),
                        config.fdv_line_volume_l,
                        places=8,
                    )
                    self.assertAlmostEqual(
                        float(row["post_fdv_inventory_l"]),
                        config.post_fdv_inventory_l,
                        places=8,
                    )

    def test_observable_and_diagnostic_counts_are_disjoint(self) -> None:
        config = quiet_config(
            duration_s=100.0,
            dt_s=0.5,
            fault_start_s=50.0,
            fault_duration_s=20.0,
        )
        records = HTSTSimulator(config).run("valve_stuck_forward_steam_loss")
        diagnostic_rows = [row for row in records if int(row["alarm_unsafe_forward"])]
        self.assertTrue(diagnostic_rows)
        for row in records:
            self.assertEqual(
                int(row["alarm_count"]),
                sum(int(row[name]) for name in ALARM_COLUMNS),
            )
            self.assertEqual(
                int(row["observable_alarm_count"]), int(row["alarm_count"])
            )
            self.assertEqual(
                int(row["diagnostic_alarm_count"]),
                sum(int(row[name]) for name in DIAGNOSTIC_ALARM_COLUMNS),
            )


class V2ConvergenceTests(unittest.TestCase):
    def test_normal_solution_converges_across_supported_steps(self) -> None:
        results: list[tuple[float, list[dict[str, float | int | str]], dict[str, object]]] = []
        for dt_s in (0.5, 0.25, 0.125):
            config = quiet_config(duration_s=160.0, dt_s=dt_s)
            records = HTSTSimulator(config).run("normal")
            results.append((dt_s, records, summarize(records, config)))

        final_temps = [float(records[-1]["holding_out_temp_c"]) for _, records, _ in results]
        energies = [float(summary["total_energy_kwh"]) for _, _, summary in results]
        forward = [float(summary["forward_l"]) for _, _, summary in results]
        self.assertLess(max(final_temps) - min(final_temps), 0.03)
        self.assertLess((max(energies) - min(energies)) / min(energies), 0.02)
        self.assertLess((max(forward) - min(forward)) / min(forward), 0.01)
        self.assertTrue(all(float(summary["unsafe_forward_l"]) == 0.0 for _, _, summary in results))


if __name__ == "__main__":
    unittest.main()
