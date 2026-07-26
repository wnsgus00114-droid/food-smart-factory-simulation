#!/usr/bin/env python3
"""State, fail-safe and cross-artifact tests for the reference PLC shadow."""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from dataclasses import fields
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from model import CampaignPhase, HTSTConfig, HTSTSimulator  # noqa: E402
from plc_logic import (  # noqa: E402
    DEFAULT_CAUSE_EFFECT_PATH,
    DEFAULT_ST_PATH,
    PLCInputs,
    PLCLogicError,
    PLCOutputs,
    PLCSettings,
    ReferencePLC,
    SUPPORTED_CAUSE_RULES,
    run_plc_shadow,
    validate_plc_traceability,
)


def good_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "time_s": 0.5,
        "step_dt_s": 0.5,
        "plant_mode": "PRODUCTION",
        "safety_temp_sensor_c": 74.0,
        "estimated_fastest_residence_time_s": 15.3,
        "measured_flow_l_h": 20_000.0,
        "maximum_safe_flow_l_h": 20_400.0,
        "measured_differential_pressure_bar": 0.80,
        "sensor_disagreement_c": 0.05,
        "leak_detector_signal_fraction": 0.0,
        "fdv_position_feedback": 0.0,
        "restart_product_interface_signal_fraction": 1.0,
        "power_good_signal": 1,
        "temperature_sensor_quality_ok": 1,
        "cip_cycle_active": 0,
        "cip_release_permissive": 1,
        "alarm_fdv_mismatch": 0,
        "alarm_regenerator_leak": 0,
    }
    row.update(changes)
    return row


def advance_to_forward(plc: ReferencePLC) -> list[dict[str, object]]:
    return [plc.scan(good_row(time_s=0.5 * index)) for index in range(1, 5)]


class ReferencePLCStateTests(unittest.TestCase):
    def test_startup_recirculation_confirmation_and_forward_sequence(self) -> None:
        plc = ReferencePLC()
        records = advance_to_forward(plc)
        self.assertEqual(
            [record["plc_state"] for record in records],
            ["STARTUP", "RECIRCULATE", "RECIRCULATE", "FORWARD"],
        )
        self.assertEqual(records[2]["forward_confirmation_elapsed_s"], 0.5)
        self.assertEqual(records[3]["forward_confirmation_elapsed_s"], 1.0)
        self.assertTrue(records[3]["output_image"]["fdv_command_forward"])
        self.assertFalse(records[3]["output_image"]["fdv_command_divert"])
        self.assertTrue(records[3]["shadow_only"])
        self.assertIn("does not drive", records[3]["assurance_note"])

    def test_each_non_latching_permissive_loss_fails_divert(self) -> None:
        cases = {
            "DIVERT_LOW_TEMPERATURE": {"safety_temp_sensor_c": 71.0},
            "DIVERT_LOW_HOLDING_TIME": {"estimated_fastest_residence_time_s": 14.0},
            "DIVERT_HIGH_FLOW": {"measured_flow_l_h": 21_000.0},
            "DIVERT_LOW_DIFFERENTIAL_PRESSURE": {
                "measured_differential_pressure_bar": 0.4
            },
            "DIVERT_SENSOR_DISAGREEMENT": {"sensor_disagreement_c": 1.0},
            "DIVERT_CIP_NOT_RELEASED": {"cip_release_permissive": 0},
        }
        for expected_cause, changes in cases.items():
            with self.subTest(cause=expected_cause):
                plc = ReferencePLC()
                self.assertEqual(advance_to_forward(plc)[-1]["plc_state"], "FORWARD")
                result = plc.scan(good_row(**changes))
                self.assertEqual(result["plc_state"], "RECIRCULATE")
                self.assertIn(expected_cause, result["active_divert_causes"].split("|"))
                self.assertFalse(result["output_image"]["fdv_command_forward"])
                self.assertTrue(result["output_image"]["fdv_command_divert"])
                self.assertEqual(result["trip_latched"], 0)

    def test_trip_is_latched_and_requires_clear_ack_and_reset(self) -> None:
        plc = ReferencePLC()
        advance_to_forward(plc)
        failed = plc.scan(good_row(power_good_signal=0))
        self.assertEqual(failed["plc_state"], "TRIP")
        self.assertEqual(failed["trip_latched"], 1)
        self.assertIn("TRIP_POWER_BAD", failed["latched_trip_causes"])
        for command, value in failed["output_image"].items():
            if command == "fdv_command_divert":
                self.assertTrue(value)
            else:
                self.assertFalse(value)

        recovered_without_ack = plc.scan(good_row(power_good_signal=1, trip_reset=1))
        self.assertEqual(recovered_without_ack["plc_state"], "TRIP")
        acknowledged = plc.scan(good_row(power_good_signal=1, trip_ack=1))
        self.assertEqual(acknowledged["trip_acknowledged"], 1)
        reset = plc.scan(good_row(power_good_signal=1, trip_reset=1))
        self.assertEqual(reset["plc_state"], "OFF")
        self.assertEqual(reset["trip_latched"], 0)
        self.assertEqual(reset["latched_trip_causes"], "")
        restarted = plc.scan(good_row())
        self.assertEqual(restarted["plc_state"], "STARTUP")

    def test_active_trip_cannot_be_reset_and_estop_is_fail_divert(self) -> None:
        plc = ReferencePLC()
        advance_to_forward(plc)
        trip = plc.scan(good_row(emergency_stop=1, trip_ack=1, trip_reset=1))
        self.assertEqual(trip["plc_state"], "TRIP")
        self.assertIn("TRIP_ESTOP", trip["active_trip_causes"])
        self.assertEqual(trip["trip_acknowledged"], 1)
        self.assertTrue(trip["output_image"]["fdv_command_divert"])
        still_active = plc.scan(good_row(emergency_stop=1, trip_reset=1))
        self.assertEqual(still_active["plc_state"], "TRIP")
        self.assertEqual(still_active["trip_latched"], 1)

    def test_operator_stop_goes_off_without_creating_a_trip(self) -> None:
        plc = ReferencePLC()
        advance_to_forward(plc)
        stopped = plc.scan(good_row(operator_stop=1))
        self.assertEqual(stopped["plc_state"], "OFF")
        self.assertEqual(stopped["trip_latched"], 0)
        self.assertTrue(stopped["output_image"]["fdv_command_divert"])
        self.assertFalse(stopped["output_image"]["feed_pump_command"])

    def test_cip_recipe_states_and_outputs_are_explicit_and_always_diverted(self) -> None:
        settings = PLCSettings(
            scan_time_s=0.5,
            cip_pre_rinse_s=1.0,
            cip_caustic_s=1.0,
            cip_intermediate_rinse_s=1.0,
            cip_acid_s=1.0,
            cip_final_rinse_s=1.0,
        )
        plc = ReferencePLC(settings)
        records = [
            plc.scan(
                good_row(
                    time_s=0.5 * index,
                    cip_cycle_active=1,
                    cip_release_permissive=0,
                    plant_mode="CIP",
                )
            )
            for index in range(1, 11)
        ]
        self.assertEqual(
            [records[index]["cip_recipe_state"] for index in (0, 1, 3, 5, 7, 9)],
            [
                "PRE_RINSE",
                "CAUSTIC",
                "INTERMEDIATE_RINSE",
                "ACID",
                "FINAL_RINSE",
                "COMPLETE",
            ],
        )
        for record in records:
            self.assertEqual(record["plc_state"], "CIP")
            self.assertFalse(record["output_image"]["fdv_command_forward"])
            self.assertTrue(record["output_image"]["fdv_command_divert"])
        self.assertTrue(records[1]["output_image"]["cip_alkali_valve"])
        self.assertTrue(records[5]["output_image"]["cip_acid_valve"])
        self.assertFalse(records[-1]["output_image"]["cip_pump_command"])
        restart = plc.scan(good_row(cip_cycle_active=0, cip_release_permissive=1))
        self.assertEqual(restart["plc_state"], "STARTUP")
        self.assertEqual(restart["cip_recipe_state"], "IDLE")

    def test_input_and_output_images_are_complete_and_rows_are_not_mutated(self) -> None:
        plc = ReferencePLC()
        row = good_row()
        original = copy.deepcopy(row)
        record = plc.scan(row)
        self.assertEqual(row, original)
        self.assertEqual(set(record["input_image"]), {item.name for item in fields(PLCInputs)})
        self.assertEqual(set(record["output_image"]), {item.name for item in fields(PLCOutputs)})

    def test_missing_nonfinite_and_out_of_range_inputs_fail_closed(self) -> None:
        missing = good_row()
        del missing["safety_temp_sensor_c"]
        with self.assertRaisesRegex(PLCLogicError, "missing 'safety_temp_sensor_c'"):
            ReferencePLC().scan(missing)
        with self.assertRaisesRegex(PLCLogicError, "must be finite"):
            ReferencePLC().scan(good_row(measured_flow_l_h=math.nan))
        with self.assertRaisesRegex(PLCLogicError, r"must be in \[0, 1\]"):
            ReferencePLC().scan(good_row(fdv_position_feedback=1.1))
        with self.assertRaisesRegex(PLCLogicError, "boolean or 0/1"):
            ReferencePLC().scan(good_row(power_good_signal=2))


class ReferencePLCIntegrationTests(unittest.TestCase):
    def test_cause_effect_python_and_structured_text_traceability(self) -> None:
        report = validate_plc_traceability()
        self.assertTrue(report["valid"])
        self.assertTrue(report["shadow_only"])
        self.assertEqual(report["cause_count"], len(SUPPORTED_CAUSE_RULES))
        matrix = json.loads(DEFAULT_CAUSE_EFFECT_PATH.read_text(encoding="utf-8"))
        source = DEFAULT_ST_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            {cause["python_rule"] for cause in matrix["causes"]},
            set(SUPPORTED_CAUSE_RULES),
        )
        for cause in matrix["causes"]:
            self.assertEqual(source.count(cause["st_marker"]), 1)

    def test_run_plc_shadow_uses_one_persistent_controller(self) -> None:
        rows = [good_row(time_s=0.5 * index) for index in range(1, 6)]
        original = copy.deepcopy(rows)
        records = run_plc_shadow(rows)
        self.assertEqual(rows, original)
        self.assertEqual(len(records), len(rows))
        self.assertEqual([item["scan_index"] for item in records], [1, 2, 3, 4, 5])
        self.assertEqual(records[3]["plc_state"], "FORWARD")
        self.assertEqual(records[4]["plc_state"], "FORWARD")

    def test_real_campaign_rows_are_accepted_without_actuating_the_simulator(self) -> None:
        config = HTSTConfig(duration_s=3.0, dt_s=0.5, random_seed=91)
        simulator = HTSTSimulator(config)
        rows = simulator.run_campaign(
            [CampaignPhase("production", "normal", 3.0)],
            campaign_id="plc-shadow-test",
        )
        final_state_before_shadow = simulator.export_state()
        records = run_plc_shadow(rows)
        final_state_after_shadow = simulator.export_state()
        self.assertEqual(len(records), len(rows))
        self.assertEqual(final_state_before_shadow, final_state_after_shadow)
        self.assertTrue(all(item["shadow_only"] for item in records))
        self.assertTrue(
            all(
                not item["output_image"]["fdv_command_forward"]
                for item in records
                if item["plc_state"] != "FORWARD"
            )
        )


if __name__ == "__main__":
    unittest.main()
