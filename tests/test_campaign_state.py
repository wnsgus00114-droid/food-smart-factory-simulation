#!/usr/bin/env python3
"""Regression tests for continuous campaign state and composition carryover."""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import (  # noqa: E402
    MODEL_VERSION,
    STATE_SCHEMA_VERSION,
    CampaignPhase,
    HTSTConfig,
    HTSTSimulator,
    summarize,
)
from process_components import (  # noqa: E402
    BalanceTank,
    BalanceTankConfig,
    LiquidStream,
)


class ProductMomentTests(unittest.TestCase):
    def test_balance_tank_preserves_product_fraction_and_state(self) -> None:
        tank = BalanceTank(
            BalanceTankConfig(
                capacity_l=100.0,
                initial_volume_l=50.0,
                minimum_operating_volume_l=10.0,
            )
        )
        step = tank.step(
            1.0,
            LiquidStream(5.0, 4.0, product_fraction=1.0),
            LiquidStream(
                5.0,
                20.0,
                chemical_fraction=0.02,
                product_fraction=0.0,
            ),
            10.0,
        )
        self.assertAlmostEqual(step.product_fraction, 55.0 / 60.0)
        self.assertAlmostEqual(step.outlet.product_fraction, 55.0 / 60.0)
        self.assertAlmostEqual(step.product_volume_error_l, 0.0, places=12)
        state = tank.export_state()
        restored = BalanceTank(tank.config)
        restored.import_state(json.loads(json.dumps(state)))
        self.assertEqual(restored.export_state(), state)


class CampaignStateTests(unittest.TestCase):
    @staticmethod
    def config(**overrides: object) -> HTSTConfig:
        values: dict[str, object] = {
            "duration_s": 120.0,
            "dt_s": 0.5,
            "fault_start_s": 20.0,
            "fault_duration_s": 60.0,
            "random_seed": 20260725,
        }
        values.update(overrides)
        return HTSTConfig(**values)

    def test_legacy_run_has_no_campaign_context_and_keeps_reference_metrics(self) -> None:
        config = HTSTConfig()
        rows = HTSTSimulator(config).run("normal")
        result = summarize(rows, config)
        self.assertEqual(MODEL_VERSION, "2.2.0")
        self.assertEqual(result["forward_l"], 4792.222)
        self.assertEqual(result["diverted_l"], 207.778)
        self.assertEqual(result["unsafe_forward_l"], 0.0)
        self.assertNotIn("campaign_id", rows[0])
        self.assertNotIn("campaign_time_s", rows[0])
        self.assertEqual(float(rows[0]["balance_tank_product_fraction"]), 1.0)

    def test_state_is_json_safe_complete_and_round_trips_exactly(self) -> None:
        simulator = HTSTSimulator(self.config())
        simulator.run_campaign(
            [CampaignPhase("production", "normal", 40.0)],
            campaign_id="roundtrip",
        )
        state = simulator.export_state()
        encoded = json.dumps(state, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        restored = HTSTSimulator(simulator.config)
        restored.import_state(decoded)
        self.assertEqual(state, restored.export_state())
        self.assertEqual(state["state_schema_version"], STATE_SCHEMA_VERSION)
        self.assertIn("rng_state", state)
        self.assertIn("holding_queue", state)
        self.assertIn("heat_exchangers", state)
        self.assertIn("cip_soil", state)

    def test_checkpoint_resume_matches_uninterrupted_campaign(self) -> None:
        config = self.config()
        phases = [
            CampaignPhase("first", "normal", 40.0),
            CampaignPhase("second", "steam_loss", 60.0),
        ]
        uninterrupted = HTSTSimulator(config)
        full_rows = uninterrupted.run_campaign(phases, campaign_id="resume")

        first = HTSTSimulator(config)
        first.run_campaign(phases[:1], campaign_id="resume")
        checkpoint = json.loads(json.dumps(first.export_state()))
        resumed = HTSTSimulator(config)
        resumed_rows = resumed.run_campaign(
            phases[1:],
            campaign_id="resume",
            initial_state=checkpoint,
        )
        expected_rows = [row for row in full_rows if row["phase_id"] == "second"]

        ignored = {"phase_index", "phase_run_id"}
        normalized_expected = [
            {key: value for key, value in row.items() if key not in ignored}
            for row in expected_rows
        ]
        normalized_resumed = [
            {key: value for key, value in row.items() if key not in ignored}
            for row in resumed_rows
        ]
        self.assertEqual(normalized_resumed, normalized_expected)
        self.assertEqual(resumed.export_state(), uninterrupted.export_state())

    def test_invalid_state_is_rejected_atomically(self) -> None:
        simulator = HTSTSimulator(self.config())
        simulator.run_phase("normal", duration_s=20.0, reset=True)
        before = simulator.export_state()
        invalid = copy.deepcopy(before)
        invalid["scalars"]["fouling_index"] = 0.9  # type: ignore[index]
        payload = dict(invalid)
        payload.pop("state_sha256")
        invalid["state_sha256"] = simulator._state_digest(payload)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            simulator.import_state(invalid)
        self.assertEqual(simulator.export_state(), before)

    def test_campaign_carries_soil_chemical_and_global_time(self) -> None:
        config = self.config(
            initial_fouling_index=0.65,
            fault_start_s=300.0,
            fault_duration_s=120.0,
        )
        phases = [
            CampaignPhase("production", "normal", 20.0),
            CampaignPhase("cip", "incomplete_cleaning", 550.0),
            CampaignPhase("restart", "normal", 180.0),
        ]
        rows = HTSTSimulator(config).run_campaign(phases, campaign_id="carry")
        by_phase = {
            phase.phase_id: [row for row in rows if row["phase_id"] == phase.phase_id]
            for phase in phases
        }
        production = by_phase["production"]
        cip = by_phase["cip"]
        restart = by_phase["restart"]

        self.assertAlmostEqual(float(rows[0]["campaign_time_start_s"]), 0.0)
        self.assertAlmostEqual(float(rows[-1]["campaign_time_s"]), 750.0)
        for left, right in zip(rows, rows[1:]):
            self.assertAlmostEqual(
                float(left["campaign_time_s"]),
                float(right["campaign_time_start_s"]),
                places=9,
            )
        self.assertAlmostEqual(
            float(production[-1]["surface_total_soil_g"]),
            float(cip[0]["surface_total_soil_g"]),
            places=9,
        )
        self.assertGreater(
            float(cip[-1]["surface_total_soil_g"]),
            config.cip_clean_soil_threshold_g,
        )
        self.assertGreater(float(restart[0]["surface_hygiene_risk_fraction"]), 0.0)
        self.assertTrue(
            any(float(row["holding_chemical_fraction"]) > 0.0 for row in restart)
        )
        self.assertTrue(
            all(
                float(row["forward_l"]) == 0.0
                for row in restart
                if int(row["cip_release_permissive"]) == 0
            )
        )
        self.assertGreater(
            sum(float(row["hygiene_noncompliant_forward_l"]) for row in restart),
            0.0,
        )
        self.assertGreaterEqual(
            sum(float(row["unsafe_forward_l"]) for row in restart),
            sum(float(row["hygiene_noncompliant_forward_l"]) for row in restart),
        )

    def test_campaign_product_tracer_displaces_line_inventory(self) -> None:
        simulator = HTSTSimulator(self.config())
        rows = simulator.run_campaign(
            [
                CampaignPhase("production", "normal", 30.0),
                CampaignPhase("rinse", "cip_cycle", 30.0),
            ],
            campaign_id="tracer",
        )
        rinse = [row for row in rows if row["phase_id"] == "rinse"]
        fractions = [float(row["routed_product_fraction"]) for row in rinse]
        self.assertGreater(max(fractions), 0.0)
        self.assertLess(fractions[-1], fractions[0])
        self.assertTrue(all(math.isfinite(value) for value in fractions))

    def test_restart_flush_drains_interface_before_product_release(self) -> None:
        config = self.config(
            initial_fouling_index=0.65,
            fault_start_s=300.0,
            fault_duration_s=120.0,
        )

        def restart_rows(cip_scenario: str) -> list[dict[str, object]]:
            rows = HTSTSimulator(config).run_campaign(
                [
                    CampaignPhase("production", "normal", 20.0),
                    CampaignPhase("cip", cip_scenario, 550.0),
                    CampaignPhase("restart", "normal", 180.0),
                ],
                campaign_id=f"flush-{cip_scenario}",
            )
            return [row for row in rows if row["phase_id"] == "restart"]

        complete = restart_rows("cip_cycle")
        incomplete = restart_rows("incomplete_cleaning")
        for rows in (complete, incomplete):
            self.assertGreater(
                sum(float(row["transition_drained_l"]) for row in rows), 0.0
            )
            self.assertGreaterEqual(
                sum(float(row["transition_drained_l"]) for row in rows),
                config.nominal_flow_l_s * config.post_fdv_residence_time_s,
            )
            self.assertEqual(
                sum(float(row["chemical_noncompliant_forward_l"]) for row in rows),
                0.0,
            )
            self.assertEqual(
                sum(float(row["dilution_noncompliant_forward_l"]) for row in rows),
                0.0,
            )
            self.assertLess(
                max(abs(float(row["post_fdv_balance_error_l"])) for row in rows),
                1e-9,
            )
            released = [row for row in rows if float(row["forward_l"]) > 0.0]
            self.assertTrue(released)
            self.assertLessEqual(
                max(float(row["potential_product_temp_c"]) for row in released),
                config.maximum_product_temp_c,
            )
        self.assertEqual(
            sum(float(row["hygiene_noncompliant_forward_l"]) for row in complete),
            0.0,
        )
        self.assertEqual(
            sum(float(row["quality_out_of_spec_l"]) for row in complete),
            0.0,
        )
        self.assertGreater(
            sum(float(row["hygiene_noncompliant_forward_l"]) for row in incomplete),
            0.0,
        )
        self.assertAlmostEqual(
            sum(float(row["quality_out_of_spec_l"]) for row in incomplete),
            sum(
                float(row["hygiene_noncompliant_forward_l"])
                for row in incomplete
            ),
        )


if __name__ == "__main__":
    unittest.main()
