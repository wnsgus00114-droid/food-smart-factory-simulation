#!/usr/bin/env python3
"""Structural contract tests for the machine-readable reference P&ID."""

from __future__ import annotations

import copy
import tempfile
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from reference_plant import (  # noqa: E402
    KNOWN_PLC_OUTPUT_FIELDS,
    KNOWN_SIMULATOR_FIELDS,
    ReferencePlantError,
    load_reference_plant,
    render_mermaid,
    validate_reference_plant,
)


class ReferencePlantContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plant = load_reference_plant()

    def test_bundled_contract_has_unique_tags_io_and_required_paths(self) -> None:
        report = validate_reference_plant(self.plant)
        self.assertTrue(report["valid"])
        self.assertTrue(report["shadow_only"])
        self.assertEqual(report["tag_count"], 61)
        self.assertEqual(report["node_count"], 22)
        self.assertEqual(report["line_count"], 23)
        self.assertEqual(report["io_count"], 26)
        self.assertEqual(
            set(report["validated_paths"]),
            {
                "RAW_TO_PRODUCT",
                "FDV_DIVERT_RETURN",
                "CIP_RECIRCULATION",
                "CIP_DRAIN",
                "CIP_ALKALI_DOSING",
                "CIP_ACID_DOSING",
            },
        )

        all_items = [
            *self.plant["boundaries"],
            *self.plant["equipment"],
            *self.plant["actuators"],
            *self.plant["lines"],
            *self.plant["instruments"],
        ]
        tags = [item["tag"] for item in all_items]
        self.assertEqual(len(tags), len(set(tags)))
        io_addresses = [item["io_address"] for item in self.plant["instruments"]]
        for actuator in self.plant["actuators"]:
            io_addresses.append(actuator["command_io"])
            if "feedback_io" in actuator:
                io_addresses.append(actuator["feedback_io"])
        self.assertEqual(len(io_addresses), len(set(io_addresses)))

    def test_every_instrument_and_actuator_has_explicit_adapter_mapping(self) -> None:
        for instrument in self.plant["instruments"]:
            self.assertIn(instrument["simulator_field"], KNOWN_SIMULATOR_FIELDS)
        for actuator in self.plant["actuators"]:
            self.assertIn(actuator["simulator_field"], KNOWN_SIMULATOR_FIELDS)
            self.assertIn(actuator["command_field"], KNOWN_PLC_OUTPUT_FIELDS)
            self.assertIn(actuator["fail_position"], {"CLOSED", "DIVERT", "OPEN", "STOPPED"})
        fdv = next(
            item
            for item in self.plant["actuators"]
            if item["kind"] == "flow_diversion_valve"
        )
        self.assertEqual(fdv["fail_position"], "DIVERT")

    def test_renderer_is_deterministic_and_github_mermaid_compatible(self) -> None:
        first = render_mermaid(self.plant)
        second = render_mermaid(copy.deepcopy(self.plant))
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("flowchart LR\n"))
        self.assertNotIn("```", first)
        for tag in (
            "RAW-INLET",
            "TK-101",
            "HT-101",
            "FDV-101",
            "PRODUCT-OUTLET",
            "CIP-201",
            "TT-104",
        ):
            self.assertIn(tag, first)
        self.assertIn("Unvalidated reference/shadow topology", first)
        self.assertIn("-. \"safety_temp_sensor_c\" .->", first)

    def test_duplicate_tag_and_io_are_rejected(self) -> None:
        duplicate_tag = copy.deepcopy(self.plant)
        duplicate_tag["equipment"][1]["tag"] = duplicate_tag["equipment"][0]["tag"]
        with self.assertRaisesRegex(ReferencePlantError, "duplicate tag"):
            validate_reference_plant(duplicate_tag)

        duplicate_io = copy.deepcopy(self.plant)
        duplicate_io["instruments"][1]["io_address"] = duplicate_io["instruments"][0]["io_address"]
        with self.assertRaisesRegex(ReferencePlantError, "duplicate I/O address"):
            validate_reference_plant(duplicate_io)

    def test_missing_required_edge_is_rejected(self) -> None:
        broken = copy.deepcopy(self.plant)
        product_line = next(item for item in broken["lines"] if item["tag"] == "L-010")
        product_line["to"] = "DRAIN-OUTLET"
        with self.assertRaisesRegex(ReferencePlantError, "RAW_TO_PRODUCT.*missing directed edges"):
            validate_reference_plant(broken)

    def test_unknown_mapping_missing_fail_position_and_non_divert_fdv_are_rejected(self) -> None:
        unknown_mapping = copy.deepcopy(self.plant)
        unknown_mapping["instruments"][0]["simulator_field"] = "imaginary_signal"
        with self.assertRaisesRegex(ReferencePlantError, "adapter contract"):
            validate_reference_plant(unknown_mapping)

        missing_fail = copy.deepcopy(self.plant)
        del missing_fail["actuators"][0]["fail_position"]
        with self.assertRaisesRegex(ReferencePlantError, "fail_position"):
            validate_reference_plant(missing_fail)

        unsafe_fdv = copy.deepcopy(self.plant)
        fdv = next(
            item
            for item in unsafe_fdv["actuators"]
            if item["kind"] == "flow_diversion_valve"
        )
        fdv["fail_position"] = "OPEN"
        with self.assertRaisesRegex(ReferencePlantError, "fail to DIVERT"):
            validate_reference_plant(unsafe_fdv)

    def test_instrument_installation_target_must_exist(self) -> None:
        broken = copy.deepcopy(self.plant)
        broken["instruments"][0]["installed_on"] = "NO-SUCH-TAG"
        with self.assertRaisesRegex(ReferencePlantError, "installed_on references unknown"):
            validate_reference_plant(broken)

    def test_every_actuator_must_be_connected_on_both_sides(self) -> None:
        broken = copy.deepcopy(self.plant)
        broken["lines"] = [
            line for line in broken["lines"] if line["from"] != "CV-201"
        ]
        with self.assertRaisesRegex(ReferencePlantError, "CV-201.*incoming and outgoing"):
            validate_reference_plant(broken)

    def test_loader_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        documents = (
            '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
            '{"schema_version":NaN}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, document in enumerate(documents):
                path = Path(temporary) / f"invalid-{index}.json"
                path.write_text(document, encoding="utf-8")
                with self.subTest(document=document):
                    with self.assertRaises(ReferencePlantError):
                        load_reference_plant(path)


if __name__ == "__main__":
    unittest.main()
