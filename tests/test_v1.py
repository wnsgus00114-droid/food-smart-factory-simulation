#!/usr/bin/env python3
"""Independent verification tests for the HTST simulator v1 API."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import (  # noqa: E402
    ALARM_COLUMNS,
    MODEL_VERSION,
    HTSTConfig,
    HTSTSimulator,
    SCENARIO_NAMES,
    ScenarioModifiers,
    extract_events,
    scenario_modifiers,
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
}


def quiet_config(**changes: float | int) -> HTSTConfig:
    values: dict[str, float | int] = dict(ZERO_NOISE)
    values.update(changes)
    return replace(HTSTConfig(), **values)


class V1ConfigurationTests(unittest.TestCase):
    def test_version_and_derived_geometry(self) -> None:
        config = HTSTConfig()
        self.assertEqual(MODEL_VERSION, "2.2.0")
        self.assertAlmostEqual(config.nominal_flow_l_s, 20_000.0 / 3600.0)
        self.assertAlmostEqual(
            config.holding_tube_volume_l,
            config.nominal_flow_l_s * config.nominal_holding_time_s,
        )
        self.assertAlmostEqual(config.maximum_safe_flow_l_h, 20_400.0)
        self.assertAlmostEqual(
            config.fdv_line_volume_l,
            config.nominal_flow_l_s * config.sensor_to_fdv_delay_s,
        )

    def test_every_numeric_setting_rejects_nonfinite_values(self) -> None:
        config = HTSTConfig()
        for name, value in asdict(config).items():
            if isinstance(value, (int, float)):
                for invalid in (math.nan, math.inf, -math.inf):
                    with self.subTest(name=name, invalid=invalid):
                        with self.assertRaises(ValueError):
                            replace(config, **{name: invalid})

    def test_strictly_positive_settings_reject_zero_and_negative_values(self) -> None:
        config = HTSTConfig()
        names = (
            "dt_s",
            "duration_s",
            "nominal_flow_l_h",
            "nominal_holding_time_s",
            "minimum_holding_time_s",
            "density_kg_l",
            "heat_capacity_kj_kg_k",
            "regenerator_tau_s",
            "heater_tau_s",
            "heater_max_delta_c",
            "lethality_reference_time_s",
            "lethality_z_c",
            "pump_efficiency",
        )
        for name in names:
            for invalid in (0.0, -0.1):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        replace(config, **{name: invalid})

    def test_bounded_and_relational_settings_are_validated(self) -> None:
        config = HTSTConfig()
        bounded = (
            "regenerator_effectiveness",
            "cooler_effectiveness",
            "pump_efficiency",
            "initial_fouling_index",
            "cip_initial_fouling_index",
            "fouling_heat_loss_fraction",
            "fouling_regeneration_loss_fraction",
        )
        for name in bounded:
            for invalid in (-0.01, 1.01):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        replace(config, **{name: invalid})

        with self.assertRaises(ValueError):
            replace(config, dt_s=1.01)
        with self.assertRaises(ValueError):
            replace(config, minimum_holding_time_s=config.nominal_holding_time_s + 0.1)
        with self.assertRaises(ValueError):
            replace(config, diversion_threshold_c=config.pasteurization_setpoint_c)
        with self.assertRaises(ValueError):
            replace(config, fault_severity=2.01)

    def test_safety_critical_thresholds_reject_unphysical_values(self) -> None:
        config = HTSTConfig()
        invalid_cases = (
            {"required_pressure_differential_bar": -0.01},
            {"sensor_disagreement_limit_c": -0.01},
            {"fouling_alarm_index": -0.01},
            {"fouling_alarm_index": 1.01},
            {"raw_side_pressure_bar": -0.01},
            {"booster_pressure_gain_bar": -0.01},
            {"pasteurized_line_drop_bar": -0.01},
            {"fouling_pressure_drop_bar": -0.01},
            {"maximum_product_temp_c": config.final_product_target_c - 0.01},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(config, **changes)


class V1TimeAndDeterminismTests(unittest.TestCase):
    def test_run_ends_exactly_and_splits_partial_and_event_steps(self) -> None:
        config = quiet_config(
            duration_s=2.25,
            dt_s=0.7,
            fault_start_s=0.9,
            fault_duration_s=0.6,
        )
        records = HTSTSimulator(config).run("steam_loss")

        expected_ends = (0.7, 0.9, 1.5, 2.2, 2.25)
        self.assertEqual(tuple(row["time_s"] for row in records), expected_ends)
        self.assertAlmostEqual(sum(float(row["step_dt_s"]) for row in records), 2.25)
        self.assertEqual(float(records[-1]["time_s"]), config.duration_s)
        for previous, current in zip(records, records[1:]):
            self.assertAlmostEqual(float(previous["time_s"]), float(current["time_start_s"]))

        active = [row for row in records if int(row["fault_active"])]
        self.assertEqual(len(active), 1)
        self.assertEqual(float(active[0]["time_start_s"]), 0.9)
        self.assertEqual(float(active[0]["time_s"]), 1.5)

    def test_unknown_scenario_is_rejected_before_running(self) -> None:
        with self.assertRaises(ValueError):
            HTSTSimulator(quiet_config(duration_s=1.0)).run("not-a-scenario")

    def test_same_instance_is_deterministic_across_repeated_runs(self) -> None:
        config = replace(HTSTConfig(), duration_s=90.0, fault_start_s=40.0)
        simulator = HTSTSimulator(config)
        first = simulator.run("control_sensor_bias_high")
        second = simulator.run("control_sensor_bias_high")
        self.assertEqual(first, second)
        self.assertEqual(summarize(first, config), summarize(second, config))

    def test_zero_severity_is_no_fault_and_compound_scalars_are_continuous(self) -> None:
        zero = replace(HTSTConfig(), fault_severity=0.0)
        for scenario in SCENARIO_NAMES:
            with self.subTest(scenario=scenario, severity=0.0):
                self.assertEqual(
                    scenario_modifiers(scenario, zero.fault_start_s + 1.0, zero),
                    ScenarioModifiers(),
                )

        half = replace(HTSTConfig(), fault_severity=0.5)
        hidden_flow = scenario_modifiers(
            "flowmeter_bias_low_with_surge",
            half.fault_start_s + 1.0,
            half,
        )
        self.assertAlmostEqual(hidden_flow.flow_factor, 1.175)
        self.assertAlmostEqual(hidden_flow.flow_meter_bias_fraction, -0.15)
        leak = scenario_modifiers(
            "regenerator_leak_pressure_inversion",
            half.fault_start_s + 1.0,
            half,
        )
        self.assertAlmostEqual(leak.booster_factor, 0.55)
        self.assertAlmostEqual(leak.leak_fraction, 0.01)
        fouling = scenario_modifiers(
            "progressive_fouling",
            half.fault_start_s + 1.0,
            half,
        )
        self.assertAlmostEqual(fouling.fouling_rate_factor, 40.5)
        stuck = scenario_modifiers(
            "valve_stuck_forward_steam_loss",
            half.fault_start_s + 1.0,
            half,
        )
        self.assertAlmostEqual(stuck.heater_capacity_factor, 0.5)
        self.assertTrue(stuck.valve_stuck_forward)


class V1TransportAndFDVTests(unittest.TestCase):
    def test_nominal_holding_residence_has_no_step_size_bias(self) -> None:
        for dt_s in (1.0, 0.5, 0.25, 0.2):
            with self.subTest(dt_s=dt_s):
                config = quiet_config(duration_s=1.0, dt_s=dt_s)
                simulator = HTSTSimulator(config)
                time_s = 0.0
                for _ in range(4):
                    volume_l = config.nominal_flow_l_s * dt_s
                    outlet = simulator._move_through_holding_tube(
                        volume_l,
                        72.0,
                        time_s,
                        time_s + dt_s,
                    )
                    self.assertAlmostEqual(
                        outlet.mean_residence_time_s,
                        config.nominal_holding_time_s,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        sum(parcel.volume_l for parcel in simulator.holding_queue),
                        config.holding_tube_volume_l,
                        places=10,
                    )
                    time_s += dt_s

    def test_holding_tube_rejects_invalid_inlet_volume(self) -> None:
        simulator = HTSTSimulator(quiet_config(duration_s=1.0))
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    simulator._move_through_holding_tube(invalid, 72.0, 0.0, 0.5)

    def test_forward_requires_line_flush_plus_confirmation_volume(self) -> None:
        config = quiet_config(duration_s=1.0, dt_s=0.5)
        simulator = HTSTSimulator(config)
        modifiers = ScenarioModifiers()
        observed_volume_l = config.nominal_flow_l_s * config.dt_s
        required_steps = math.ceil(
            (config.fdv_line_volume_l + config.nominal_flow_l_s * config.forward_confirmation_s)
            / observed_volume_l
            - 1e-12
        )

        for _ in range(required_steps - 1):
            simulator._update_fdv(True, modifiers, config.dt_s, observed_volume_l)
            self.assertFalse(simulator.fdv_forward)
        simulator._update_fdv(True, modifiers, config.dt_s, observed_volume_l)
        self.assertTrue(simulator.forward_permissive_confirmed)
        simulator._update_fdv(True, modifiers, config.dt_s, observed_volume_l)
        self.assertTrue(simulator.fdv_forward)

    def test_bad_permissive_resets_forward_flush_and_divert_delay_is_bounded(self) -> None:
        config = quiet_config(duration_s=1.0, dt_s=0.1, divert_actuation_delay_s=0.2)
        simulator = HTSTSimulator(config)
        modifiers = ScenarioModifiers()
        volume_l = config.nominal_flow_l_s * config.dt_s

        simulator._update_fdv(True, modifiers, config.dt_s, volume_l)
        self.assertGreater(simulator.pending_forward_safe_volume_l, 0.0)
        simulator._update_fdv(False, modifiers, config.dt_s, 0.0)
        self.assertEqual(simulator.pending_forward_safe_volume_l, 0.0)

        simulator.fdv_forward = True
        required_closing_steps = math.ceil(
            (config.divert_actuation_delay_s + config.fdv_travel_time_s)
            / config.dt_s
            - 1e-12
        )
        for _ in range(required_closing_steps - 1):
            simulator._update_fdv(False, modifiers, config.dt_s, 0.0)
            self.assertTrue(simulator.fdv_forward)
        simulator._update_fdv(False, modifiers, config.dt_s, 0.0)
        self.assertFalse(simulator.fdv_forward)

        simulator.fdv_forward = True
        simulator._update_fdv(
            True,
            ScenarioModifiers(power_available=False),
            config.dt_s,
            volume_l,
        )
        self.assertFalse(simulator.fdv_forward)


class V1ScenarioFixture(unittest.TestCase):
    config: HTSTConfig
    records_by_scenario: dict[str, list[dict[str, float | int | str]]]
    summaries: dict[str, dict[str, object]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = quiet_config(
            duration_s=180.0,
            dt_s=0.5,
            fault_start_s=80.0,
            fault_duration_s=50.0,
        )
        cls.records_by_scenario = {
            scenario: HTSTSimulator(cls.config).run(scenario)
            for scenario in SCENARIO_NAMES
        }
        cls.summaries = {
            scenario: summarize(records, cls.config)
            for scenario, records in cls.records_by_scenario.items()
        }


class V1BalanceAndInvariantTests(V1ScenarioFixture):
    def test_inventory_and_material_balance_for_every_scenario(self) -> None:
        config = self.config
        for scenario, records in self.records_by_scenario.items():
            expected_feed_l = 0.0
            total_routed_l = 0.0
            for index, row in enumerate(records):
                with self.subTest(scenario=scenario, index=index):
                    dt_s = float(row["step_dt_s"])
                    feed_l = float(row["flow_l_h"]) / 3600.0 * dt_s
                    expected_feed_l += feed_l
                    total_routed_l += float(row["routed_volume_l"])
                    self.assertAlmostEqual(
                        float(row["routed_volume_l"]),
                        float(row["forward_l"])
                        + float(row["diverted_l"])
                        + float(row["cip_recirculated_l"]),
                        places=10,
                    )
                    self.assertAlmostEqual(float(row["mass_balance_error_l"]), 0.0, places=10)
                    self.assertAlmostEqual(
                        float(row["holding_inventory_l"]),
                        config.holding_tube_volume_l,
                        places=9,
                    )
                    self.assertAlmostEqual(
                        float(row["fdv_line_inventory_l"]),
                        config.fdv_line_volume_l,
                        places=9,
                    )
                    self.assertLessEqual(float(row["unsafe_forward_l"]), float(row["forward_l"]))
                    self.assertAlmostEqual(
                        float(row["safe_forward_l"])
                        + float(row["unsafe_forward_l"]),
                        float(row["forward_l"]),
                        places=10,
                    )
                    for component in (
                        "thermal_unsafe_forward_l",
                        "pressure_noncompliant_forward_l",
                        "contamination_exposed_forward_l",
                    ):
                        self.assertLessEqual(
                            float(row[component]),
                            float(row["unsafe_forward_l"]) + 1e-10,
                        )
                    self.assertEqual(
                        int(row["alarm_count"]),
                        sum(int(row[name]) for name in ALARM_COLUMNS),
                    )
            with self.subTest(scenario=scenario, balance="cumulative-volume"):
                self.assertAlmostEqual(total_routed_l, expected_feed_l, places=8)
                self.assertAlmostEqual(
                    total_routed_l * config.density_kg_l,
                    expected_feed_l * config.density_kg_l,
                    places=8,
                )

    def test_summary_reports_zero_balance_and_inventory_errors(self) -> None:
        for scenario, summary in self.summaries.items():
            with self.subTest(scenario=scenario):
                self.assertLessEqual(float(summary["maximum_abs_mass_balance_error_l"]), 1e-10)
                self.assertLessEqual(float(summary["maximum_holding_inventory_error_l"]), 1e-8)
                self.assertLessEqual(float(summary["maximum_fdv_line_inventory_error_l"]), 1e-8)
                self.assertLessEqual(float(summary["maximum_post_fdv_inventory_error_l"]), 1e-8)
                self.assertLessEqual(
                    abs(float(summary["external_volume_balance_error_l"])), 1e-8
                )


class V1SensorAndHydraulicInterlockTests(V1ScenarioFixture):
    def _fault_rows(self, scenario: str) -> list[dict[str, float | int | str]]:
        rows = [row for row in self.records_by_scenario[scenario] if int(row["fault_active"])]
        self.assertTrue(rows, f"no active fault rows for {scenario}")
        return rows

    def test_each_independent_temperature_sensor_bias_fails_safe(self) -> None:
        for scenario in ("control_sensor_bias_high", "safety_sensor_bias_high"):
            rows = self._fault_rows(scenario)
            with self.subTest(scenario=scenario):
                self.assertTrue(all(int(row["alarm_sensor_disagreement"]) for row in rows))
                self.assertTrue(all(not int(row["fdv_command_forward"]) for row in rows))
                self.assertLessEqual(
                    sum(float(row["forward_l"]) for row in rows),
                    self.config.nominal_flow_l_s
                    * (
                        self.config.divert_actuation_delay_s
                        + 0.5 * self.config.fdv_travel_time_s
                    )
                    + 1e-9,
                )
                self.assertTrue(all(float(row["forward_l"]) == 0.0 for row in rows[1:]))
                self.assertEqual(float(self.summaries[scenario]["unsafe_forward_l"]), 0.0)

    def test_common_mode_sensor_bias_remains_a_detectable_modelled_hazard(self) -> None:
        rows = self._fault_rows("dual_sensor_common_bias")
        self.assertTrue(all(not int(row["alarm_sensor_disagreement"]) for row in rows))
        self.assertGreater(sum(float(row["forward_l"]) for row in rows), 0.0)
        self.assertGreater(float(self.summaries["dual_sensor_common_bias"]["unsafe_forward_l"]), 0.0)

    def test_booster_failure_trips_pressure_permissive_and_diverts(self) -> None:
        rows = self._fault_rows("booster_pump_failure")
        self.assertTrue(all(float(row["booster_pump_factor"]) == 0.0 for row in rows))
        self.assertTrue(all(int(row["alarm_low_pressure_differential"]) for row in rows))
        self.assertTrue(all(not int(row["fdv_command_forward"]) for row in rows))
        self.assertLessEqual(
            sum(float(row["forward_l"]) for row in rows),
            self.config.nominal_flow_l_s
            * (
                self.config.divert_actuation_delay_s
                + 0.5 * self.config.fdv_travel_time_s
            )
            + 1e-6,
        )
        self.assertTrue(all(float(row["forward_l"]) == 0.0 for row in rows[1:]))
        self.assertGreater(
            float(self.summaries["booster_pump_failure"]["unsafe_forward_l"]),
            0.0,
        )
        self.assertLessEqual(
            float(self.summaries["booster_pump_failure"]["unsafe_forward_l"]),
            self.config.nominal_flow_l_s
            * (
                self.config.divert_actuation_delay_s
                + 0.5 * self.config.fdv_travel_time_s
            )
            + 1e-6,
        )

    def test_regenerator_leak_and_pressure_inversion_block_forward_flow(self) -> None:
        rows = self._fault_rows("regenerator_leak_pressure_inversion")
        self.assertTrue(all(float(row["leak_fraction"]) > 0.0 for row in rows))
        self.assertTrue(all(int(row["contamination_risk"]) for row in rows))
        self.assertTrue(all(int(row["alarm_regenerator_leak"]) for row in rows))
        self.assertTrue(all(int(row["alarm_low_pressure_differential"]) for row in rows))
        self.assertTrue(all(not int(row["fdv_command_forward"]) for row in rows))
        self.assertLessEqual(
            sum(float(row["forward_l"]) for row in rows),
            self.config.nominal_flow_l_s
            * (
                self.config.divert_actuation_delay_s
                + 0.5 * self.config.fdv_travel_time_s
            )
            + 1e-9,
        )
        self.assertTrue(all(float(row["forward_l"]) == 0.0 for row in rows[1:]))
        self.assertGreater(
            float(self.summaries["regenerator_leak_pressure_inversion"]["unsafe_forward_l"]),
            0.0,
        )
        self.assertLessEqual(
            float(self.summaries["regenerator_leak_pressure_inversion"]["unsafe_forward_l"]),
            self.config.nominal_flow_l_s
            * (
                self.config.divert_actuation_delay_s
                + 0.5 * self.config.fdv_travel_time_s
            )
            + 1e-6,
        )


class V1CIPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = quiet_config(duration_s=550.0, dt_s=1.0)
        cls.records = HTSTSimulator(cls.config).run("cip_cycle")

    def test_cip_phase_schedule_and_routing(self) -> None:
        expected_durations = {
            "CIP_PRE_RINSE": 60.0,
            "CIP_CAUSTIC": 180.0,
            "CIP_INTERMEDIATE_RINSE": 60.0,
            "CIP_ACID": 120.0,
            "CIP_FINAL_RINSE": 120.0,
            "CIP_COMPLETE": 10.0,
        }
        actual_durations: dict[str, float] = {}
        for row in self.records:
            mode = str(row["plant_mode"])
            actual_durations[mode] = actual_durations.get(mode, 0.0) + float(row["step_dt_s"])
            self.assertEqual(float(row["forward_l"]), 0.0)
            self.assertEqual(float(row["diverted_l"]), 0.0)
            self.assertAlmostEqual(
                float(row["routed_volume_l"]),
                float(row["cip_recirculated_l"]),
                places=10,
            )
            self.assertEqual(int(row["fdv_command_forward"]), 0)
            self.assertEqual(int(row["fdv_actual_forward"]), 0)
        self.assertEqual(actual_durations, expected_durations)

    def test_caustic_and_acid_reduce_fouling_monotonically(self) -> None:
        fouling = [float(row["fouling_index"]) for row in self.records]
        self.assertTrue(all(after <= before + 1e-12 for before, after in zip(fouling, fouling[1:])))
        self.assertAlmostEqual(fouling[0], self.config.cip_initial_fouling_index)
        self.assertLess(fouling[-1], 0.20)

        for mode in ("CIP_CAUSTIC", "CIP_ACID"):
            phase_values = [
                float(row["fouling_index"])
                for row in self.records
                if row["plant_mode"] == mode
            ]
            with self.subTest(mode=mode):
                self.assertGreater(phase_values[0], phase_values[-1])


class V1EventTests(unittest.TestCase):
    def test_event_extraction_emits_only_mode_and_alarm_edges(self) -> None:
        base = {name: 0 for name in ALARM_COLUMNS}
        rows = [
            {**base, "time_s": 0.5, "plant_mode": "STARTUP", "alarm_low_temperature": 1},
            {**base, "time_s": 1.0, "plant_mode": "STARTUP", "alarm_high_flow": 1},
            {**base, "time_s": 1.5, "plant_mode": "DIVERT", "alarm_high_flow": 1},
            {**base, "time_s": 2.0, "plant_mode": "DIVERT"},
        ]
        self.assertEqual(
            extract_events(rows),
            [
                {"time_s": 0.5, "event_type": "MODE", "event_name": "STARTUP", "state": "ENTERED"},
                {
                    "time_s": 0.5,
                    "event_type": "ALARM",
                    "event_name": "alarm_low_temperature",
                    "state": "ACTIVE",
                },
                {
                    "time_s": 1.0,
                    "event_type": "ALARM",
                    "event_name": "alarm_low_temperature",
                    "state": "CLEARED",
                },
                {
                    "time_s": 1.0,
                    "event_type": "ALARM",
                    "event_name": "alarm_high_flow",
                    "state": "ACTIVE",
                },
                {"time_s": 1.5, "event_type": "MODE", "event_name": "DIVERT", "state": "ENTERED"},
                {
                    "time_s": 2.0,
                    "event_type": "ALARM",
                    "event_name": "alarm_high_flow",
                    "state": "CLEARED",
                },
            ],
        )

    def test_pressure_alarm_has_one_activation_and_one_clear_event(self) -> None:
        config = quiet_config(
            duration_s=150.0,
            dt_s=0.5,
            fault_start_s=70.0,
            fault_duration_s=30.0,
        )
        records = HTSTSimulator(config).run("booster_pump_failure")
        edges = [
            event
            for event in extract_events(records)
            if event["event_name"] == "alarm_low_pressure_differential"
        ]
        self.assertEqual([event["state"] for event in edges], ["ACTIVE", "CLEARED"])
        self.assertGreaterEqual(float(edges[0]["time_s"]), config.fault_start_s)
        self.assertGreaterEqual(
            float(edges[1]["time_s"]),
            config.fault_start_s + config.fault_duration_s,
        )


class V1ScenarioContractTests(V1ScenarioFixture):
    def test_normal_and_detectable_single_faults_do_not_release_unsafe_product(self) -> None:
        fail_safe_scenarios = (
            "normal",
            "steam_loss",
            "flow_surge",
            "control_sensor_bias_high",
            "safety_sensor_bias_high",
            "cooling_utility_loss",
            "power_failure",
        )
        for scenario in fail_safe_scenarios:
            with self.subTest(scenario=scenario):
                self.assertGreater(float(self.summaries[scenario]["forward_l"]), 0.0)
                self.assertEqual(float(self.summaries[scenario]["unsafe_forward_l"]), 0.0)

        for scenario in (
            "booster_pump_failure",
            "regenerator_leak_pressure_inversion",
        ):
            with self.subTest(transient_closure=scenario):
                self.assertGreater(
                    float(self.summaries[scenario]["unsafe_forward_l"]), 0.0
                )
                self.assertLessEqual(
                    float(self.summaries[scenario]["unsafe_forward_l"]),
                    self.config.nominal_flow_l_s
                    * (
                        self.config.divert_actuation_delay_s
                        + 0.5 * self.config.fdv_travel_time_s
                    )
                    + 1e-6,
                )

        self.assertGreater(
            float(self.summaries["steam_loss"]["diverted_l"]),
            float(self.summaries["normal"]["diverted_l"]),
        )
        self.assertGreater(
            float(self.summaries["flow_surge"]["diverted_l"]),
            float(self.summaries["normal"]["diverted_l"]),
        )

    def test_latent_and_compound_faults_produce_modelled_unsafe_release(self) -> None:
        for scenario in (
            "dual_sensor_common_bias",
            "flowmeter_bias_low_with_surge",
            "valve_stuck_forward_steam_loss",
        ):
            with self.subTest(scenario=scenario):
                self.assertGreater(float(self.summaries[scenario]["unsafe_forward_l"]), 0.0)
                self.assertGreater(
                    float(self.summaries[scenario]["unsafe_forward_fraction_of_forward"]),
                    0.0,
                )

        hidden_flow_rows = [
            row
            for row in self.records_by_scenario["flowmeter_bias_low_with_surge"]
            if int(row["fault_active"])
        ]
        self.assertTrue(all(not int(row["alarm_high_flow"]) for row in hidden_flow_rows))
        self.assertTrue(all(not int(row["alarm_low_holding_time"]) for row in hidden_flow_rows))

    def test_quality_and_fouling_faults_affect_their_intended_outputs(self) -> None:
        self.assertGreater(float(self.summaries["cooling_utility_loss"]["quality_out_of_spec_l"]), 0.0)
        self.assertEqual(float(self.summaries["normal"]["quality_out_of_spec_l"]), 0.0)
        self.assertGreater(
            float(self.summaries["progressive_fouling"]["final_fouling_index"]),
            float(self.summaries["normal"]["final_fouling_index"]),
        )

    def test_fault_marker_is_limited_to_the_configured_window(self) -> None:
        for scenario, records in self.records_by_scenario.items():
            if scenario in {"normal", "cip_cycle"}:
                self.assertTrue(all(not int(row["fault_active"]) for row in records))
                continue
            active = [row for row in records if int(row["fault_active"])]
            with self.subTest(scenario=scenario):
                self.assertAlmostEqual(
                    sum(float(row["step_dt_s"]) for row in active),
                    self.config.fault_duration_s,
                )
                self.assertEqual(float(active[0]["time_start_s"]), self.config.fault_start_s)
                self.assertEqual(
                    float(active[-1]["time_s"]),
                    self.config.fault_start_s + self.config.fault_duration_s,
                )


class V1StepConvergenceTests(unittest.TestCase):
    def test_normal_solution_is_step_size_consistent(self) -> None:
        results: list[tuple[HTSTConfig, list[dict[str, float | int | str]], dict[str, object]]] = []
        for dt_s in (0.5, 0.25, 0.125):
            config = quiet_config(
                duration_s=160.0,
                dt_s=dt_s,
                fault_start_s=80.0,
                fault_duration_s=30.0,
            )
            records = HTSTSimulator(config).run("normal")
            results.append((config, records, summarize(records, config)))

        final_temperatures = [float(records[-1]["holding_out_temp_c"]) for _, records, _ in results]
        total_energies = [float(summary["total_energy_kwh"]) for _, _, summary in results]
        forward_volumes = [float(summary["forward_l"]) for _, _, summary in results]
        first_forward_times = [float(summary["first_forward_time_s"]) for _, _, summary in results]

        self.assertLess(max(final_temperatures) - min(final_temperatures), 0.05)
        self.assertLess(
            (max(total_energies) - min(total_energies)) / min(total_energies),
            0.02,
        )
        self.assertLess(
            (max(forward_volumes) - min(forward_volumes)) / min(forward_volumes),
            0.02,
        )
        self.assertLessEqual(max(first_forward_times) - min(first_forward_times), 0.5)

        expected_routed_l = results[0][0].nominal_flow_l_h / 3600.0 * 160.0
        for config, records, summary in results:
            with self.subTest(dt_s=config.dt_s):
                self.assertEqual(float(records[-1]["time_s"]), config.duration_s)
                self.assertAlmostEqual(float(summary["routed_l"]), expected_routed_l, places=3)
                self.assertEqual(float(summary["unsafe_forward_l"]), 0.0)
                self.assertTrue(
                    all(
                        math.isclose(
                            float(row["actual_residence_time_s"]),
                            config.nominal_holding_time_s,
                            abs_tol=1e-9,
                        )
                        for row in records
                    )
                )


if __name__ == "__main__":
    unittest.main()
