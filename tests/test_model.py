#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import HTSTConfig, HTSTSimulator, scenario_modifiers, summarize  # noqa: E402


class HTSTModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(HTSTConfig(), duration_s=600.0, dt_s=0.5)

    def run_summary(self, scenario: str) -> dict[str, object]:
        simulator = HTSTSimulator(self.config)
        records = simulator.run(scenario)
        return summarize(records, self.config)

    def test_holding_volume_matches_nominal_flow_and_time(self) -> None:
        expected = self.config.nominal_flow_l_s * self.config.nominal_holding_time_s
        self.assertAlmostEqual(self.config.holding_tube_volume_l, expected)

    def test_flow_surge_modifier(self) -> None:
        normal = scenario_modifiers("flow_surge", 299.0)
        fault = scenario_modifiers("flow_surge", 320.0)
        self.assertEqual(normal.flow_factor, 1.0)
        self.assertEqual(fault.flow_factor, 1.5)

    def test_normal_scenario_eventually_forwards_without_unsafe_release(self) -> None:
        result = self.run_summary("normal")
        self.assertGreater(float(result["forward_l"]), 0.0)
        self.assertEqual(float(result["unsafe_forward_l"]), 0.0)
        self.assertGreaterEqual(
            float(result["minimum_forward_holding_temp_c"]),
            self.config.diversion_threshold_c,
        )

    def test_steam_loss_diverts_without_unsafe_release(self) -> None:
        result = self.run_summary("steam_loss")
        normal = self.run_summary("normal")
        self.assertGreater(float(result["diverted_fraction"]), float(normal["diverted_fraction"]))
        self.assertEqual(float(result["unsafe_forward_l"]), 0.0)

    def test_stuck_forward_combined_fault_releases_unsafe_product(self) -> None:
        result = self.run_summary("valve_stuck_forward_steam_loss")
        self.assertGreater(float(result["unsafe_forward_l"]), 0.0)

    def test_seed_makes_simulation_deterministic(self) -> None:
        first = self.run_summary("sensor_bias_high")
        second = self.run_summary("sensor_bias_high")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
