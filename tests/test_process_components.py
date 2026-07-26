#!/usr/bin/env python3
"""Independent conservation and convergence tests for v2 process components."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from process_components import (  # noqa: E402
    BalanceTank,
    BalanceTankConfig,
    CIPSoilConfig,
    CIPSoilModel,
    DynamicHeatExchanger,
    HeatExchangerConfig,
    LiquidStream,
)


class BalanceTankTests(unittest.TestCase):
    def test_mixing_and_attribute_balances_close(self) -> None:
        tank = BalanceTank(
            BalanceTankConfig(
                capacity_l=1_000.0,
                initial_volume_l=500.0,
                minimum_operating_volume_l=50.0,
                initial_temperature_c=4.0,
                ambient_temperature_c=4.0,
            )
        )
        result = tank.step(
            1.0,
            LiquidStream(50.0, 4.0),
            LiquidStream(50.0, 74.0, 2.0, 0.20, 0.04),
            100.0,
        )
        self.assertAlmostEqual(result.outlet.temperature_c, (500 * 4 + 50 * 4 + 50 * 74) / 600)
        self.assertAlmostEqual(result.outlet.mean_pass_count, 100 / 600)
        self.assertAlmostEqual(result.outlet.risk_fraction, 10 / 600)
        self.assertAlmostEqual(result.outlet.chemical_fraction, 2 / 600)
        for error in (
            result.volume_balance_error_l,
            result.temperature_moment_error_l_c,
            result.pass_moment_error_l,
            result.risk_volume_error_l,
            result.chemical_volume_error_l,
        ):
            self.assertAlmostEqual(error, 0.0, places=10)

    def test_overflow_and_starvation_fail_fast(self) -> None:
        tank = BalanceTank(BalanceTankConfig(capacity_l=100.0, initial_volume_l=80.0))
        with self.assertRaises(ValueError):
            tank.step(1.0, LiquidStream(30.0, 4.0), LiquidStream(0.0, 4.0), 0.0)
        tank.reset()
        with self.assertRaises(ValueError):
            tank.step(1.0, LiquidStream(0.0, 4.0), LiquidStream(0.0, 4.0), 40.0)

    def test_config_rejects_initial_inventory_below_operating_minimum(self) -> None:
        with self.assertRaises(ValueError):
            BalanceTankConfig(
                capacity_l=100.0,
                initial_volume_l=40.0,
                minimum_operating_volume_l=50.0,
            )


class DynamicHeatExchangerTests(unittest.TestCase):
    def _run(self, dt: float) -> tuple[float, float, float]:
        exchanger = DynamicHeatExchanger(
            HeatExchangerConfig(internal_max_dt_s=min(0.02, dt)),
            initial_product_c=4.0,
            initial_utility_c=80.0,
        )
        elapsed = 0.0
        maximum_error = 0.0
        while elapsed < 10.0 - 1e-12:
            step = exchanger.step(
                min(dt, 10.0 - elapsed),
                product_inlet_c=4.0,
                product_flow_l_s=5.0,
                utility_inlet_c=80.0,
                utility_flow_l_s=5.0,
                fouling_index=0.2,
            )
            maximum_error = max(maximum_error, abs(step.energy_balance_error_kj))
            elapsed += min(dt, 10.0 - elapsed)
        return step.product_outlet_c, step.utility_outlet_c, maximum_error

    def test_energy_balance_and_dt_convergence(self) -> None:
        coarse = self._run(0.5)
        fine = self._run(0.125)
        self.assertLess(coarse[2], 1e-8)
        self.assertLess(fine[2], 1e-8)
        self.assertAlmostEqual(coarse[0], fine[0], delta=0.05)
        self.assertAlmostEqual(coarse[1], fine[1], delta=0.05)

    def test_fouling_reduces_effective_ua_and_zero_flow_cools(self) -> None:
        clean = DynamicHeatExchanger(initial_product_c=70.0, initial_utility_c=70.0, initial_wall_c=70.0)
        clean_step = clean.step(
            1.0,
            product_inlet_c=70.0,
            product_flow_l_s=0.0,
            utility_inlet_c=70.0,
            utility_flow_l_s=0.0,
            fouling_index=0.0,
        )
        fouled = DynamicHeatExchanger()
        fouled_step = fouled.step(
            1.0,
            product_inlet_c=4.0,
            product_flow_l_s=5.0,
            utility_inlet_c=80.0,
            utility_flow_l_s=5.0,
            fouling_index=1.0,
        )
        self.assertLess(fouled_step.effective_ua_kw_k, clean_step.effective_ua_kw_k)
        self.assertLess(clean_step.wall_temperature_c, 70.0)

    def test_invalid_or_nonfinite_inputs_are_rejected(self) -> None:
        exchanger = DynamicHeatExchanger()
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    exchanger.step(
                        1.0,
                        product_inlet_c=invalid,
                        product_flow_l_s=1.0,
                        utility_inlet_c=80.0,
                        utility_flow_l_s=1.0,
                    )


class CIPSoilTests(unittest.TestCase):
    def test_config_rejects_clean_threshold_above_initial_soil(self) -> None:
        with self.assertRaises(ValueError):
            CIPSoilConfig(
                initial_protein_soil_g=5.0,
                initial_mineral_soil_g=5.0,
                clean_soil_threshold_g=11.0,
            )

    def test_alkali_and_acid_remove_separate_soils_monotonically(self) -> None:
        model = CIPSoilModel()
        first = model.step(
            60.0,
            temperature_c=75.0,
            velocity_m_s=1.5,
            alkali_pct=1.0,
            rinse_flow_l_s=2.0,
        )
        self.assertLess(first.protein_soil_g, model.config.initial_protein_soil_g)
        self.assertEqual(first.mineral_soil_g, model.config.initial_mineral_soil_g)
        second = model.step(
            60.0,
            temperature_c=65.0,
            velocity_m_s=1.5,
            acid_pct=0.8,
            rinse_flow_l_s=2.0,
        )
        self.assertLess(second.mineral_soil_g, first.mineral_soil_g)
        self.assertLessEqual(second.protein_soil_g, first.protein_soil_g)

    def test_incomplete_cycle_stays_incomplete_and_rinse_removes_residual(self) -> None:
        model = CIPSoilModel(CIPSoilConfig(clean_soil_threshold_g=100.0))
        incomplete = model.step(
            10.0,
            temperature_c=30.0,
            velocity_m_s=0.2,
            alkali_pct=0.2,
            rinse_flow_l_s=1.0,
        )
        self.assertFalse(incomplete.cleaning_complete)
        before = incomplete.residual_chemical_fraction
        for _ in range(20):
            rinsed = model.step(
                10.0,
                temperature_c=20.0,
                velocity_m_s=1.0,
                rinse_flow_l_s=10.0,
            )
        self.assertLess(rinsed.residual_chemical_fraction, before)

    def test_rinse_displacement_is_timestep_invariant(self) -> None:
        def run(dt_s: float) -> float:
            model = CIPSoilModel()
            model.step(
                10.0,
                temperature_c=70.0,
                velocity_m_s=1.5,
                alkali_pct=2.0,
                rinse_flow_l_s=10.0,
            )
            elapsed = 0.0
            while elapsed < 20.0 - 1e-12:
                step = min(dt_s, 20.0 - elapsed)
                result = model.step(
                    step,
                    temperature_c=20.0,
                    velocity_m_s=1.5,
                    rinse_flow_l_s=10.0,
                )
                elapsed += step
            return result.residual_chemical_fraction

        self.assertAlmostEqual(run(10.0), run(0.25), places=12)


if __name__ == "__main__":
    unittest.main()
