#!/usr/bin/env python3
"""Contract tests for the independent reference HACCP evidence layer."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from haccp import (  # noqa: E402
    EvidenceError,
    EvidenceLedger,
    GENESIS_HASH,
    HACCPPlanError,
    MANDATORY_SOURCE_IDS,
    REQUIRED_RUNTIME_SIGNALS,
    load_haccp_plan,
    validate_haccp_plan,
    verify_evidence_chain,
)
from model import HTSTConfig, HTSTSimulator  # noqa: E402


VALID_CALIBRATIONS: dict[str, object] = {
    "TT-104": "VALID",
    "FT-101": "VALID",
    "PT-101": "VALID",
    "PT-102": "VALID",
    "PDT-101": "VALID",
    "AIT-201": "VALID",
    "AIT-202": "VALID",
    "AIT-203": "VALID",
}


def good_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "time_s": 0.5,
        "cip_cycle_active": 0,
        "safety_temp_sensor_c": 73.0,
        "estimated_fastest_residence_time_s": 15.3,
        "measured_flow_l_h": 20_000.0,
        "maximum_safe_flow_l_h": 20_400.0,
        "measured_differential_pressure_bar": 0.8,
        "cip_release_permissive": 1,
    }
    row.update(changes)
    return row


def records_of(result: dict[str, object], record_type: str) -> list[dict[str, object]]:
    records = result["records_appended"]
    assert isinstance(records, list)
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("record_type") == record_type
    ]


class HACCPPlanContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_haccp_plan()

    def test_plan_covers_sources_hazards_controls_and_five_runtime_limits(self) -> None:
        report = validate_haccp_plan(self.plan)
        self.assertTrue(report["valid"])
        self.assertEqual(report["conformity_status"], "not_assessed")
        self.assertTrue(report["shadow_only"])
        self.assertEqual(report["source_count"], 3)
        self.assertEqual(report["hazard_count"], 4)
        self.assertEqual(report["control_count"], 3)
        self.assertEqual(report["critical_limit_count"], 5)
        self.assertEqual(set(report["runtime_signals"]), set(REQUIRED_RUNTIME_SIGNALS))

        sources = {item["source_id"]: item for item in self.plan["source_registry"]}
        self.assertEqual(set(sources), set(MANDATORY_SOURCE_IDS))
        for source in sources.values():
            self.assertEqual(source["conformity_status"], "not_assessed")
            self.assertEqual(
                [principle["number"] for principle in source["principles"]],
                list(range(1, 8)),
            )
        korean = sources["KR-MFDS-2026-25-ART6"]
        self.assertIn("제2026-25호", korean["identifier"])
        self.assertIn("제6조", korean["provision"])
        self.assertEqual(
            {item["category"] for item in self.plan["hazard_analysis"]},
            {"PROCESS_CONTROL", "CHEMICAL", "PHYSICAL"},
        )
        self.assertEqual(
            {item["control_type"] for item in self.plan["controls"]},
            {"CCP", "OPRP"},
        )

    def test_plan_rejects_conformity_certification_and_automatic_release_claims(self) -> None:
        mutations: list[tuple[str, dict[str, object], str]] = []

        assessed = copy.deepcopy(self.plan)
        assessed["conformity_status"] = "certified"
        mutations.append(("conformity", assessed, "not_assessed"))

        certification = copy.deepcopy(self.plan)
        certification["assurance"]["certification_claim"] = True
        mutations.append(("certification", certification, "must be false"))

        auto_release = copy.deepcopy(self.plan)
        auto_release["controls"][0]["corrective_action"]["automatic_release"] = True
        mutations.append(("automatic release", auto_release, "forbid automatic release"))

        for label, mutation, message in mutations:
            with self.subTest(label=label):
                with self.assertRaisesRegex(HACCPPlanError, message):
                    validate_haccp_plan(mutation)

    def test_plan_rejects_incomplete_seven_principle_and_control_fields(self) -> None:
        missing_principle = copy.deepcopy(self.plan)
        missing_principle["source_registry"][0]["principles"].pop()

        missing_rationale = copy.deepcopy(self.plan)
        missing_rationale["hazard_analysis"][0]["rationale"] = ""

        missing_monitor_who = copy.deepcopy(self.plan)
        del missing_monitor_who["controls"][0]["monitoring"]["who"]

        missing_verification = copy.deepcopy(self.plan)
        missing_verification["controls"][0]["verification"]["activities"] = []

        missing_records = copy.deepcopy(self.plan)
        missing_records["controls"][0]["records"].remove("DEVIATION_RECORD")

        mutations = [
            (missing_principle, "seven principles"),
            (missing_rationale, "rationale"),
            (missing_monitor_who, "who"),
            (missing_verification, "non-empty array"),
            (missing_records, "missing required evidence types"),
        ]
        for mutation, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(HACCPPlanError, message):
                    validate_haccp_plan(mutation)

    def test_plan_rejects_ambiguous_limits_and_broken_traceability(self) -> None:
        ambiguous = copy.deepcopy(self.plan)
        ambiguous["controls"][0]["critical_limits"][0][
            "reference_signal"
        ] = "maximum_safe_flow_l_h"

        duplicate_signal = copy.deepcopy(self.plan)
        duplicate_signal["controls"][0]["critical_limits"][0]["signal"] = (
            "estimated_fastest_residence_time_s"
        )

        bad_hazard_reference = copy.deepcopy(self.plan)
        bad_hazard_reference["hazard_analysis"][0]["control_ids"] = ["UNKNOWN-CCP"]

        one_way_reference = copy.deepcopy(self.plan)
        one_way_reference["hazard_analysis"][0]["control_ids"] = []
        one_way_reference["hazard_analysis"][0]["scope_status"] = (
            "outside_runtime_scope_not_assessed"
        )

        mutations = [
            (ambiguous, "exactly one"),
            (duplicate_signal, "five required signals"),
            (bad_hazard_reference, "unknown controls"),
            (one_way_reference, "not bidirectional"),
        ]
        for mutation, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(HACCPPlanError, message):
                    validate_haccp_plan(mutation)

    def test_plan_json_and_schema_validation_are_strict(self) -> None:
        extra = copy.deepcopy(self.plan)
        extra["unexpected"] = True
        with self.assertRaisesRegex(HACCPPlanError, "keys mismatch"):
            validate_haccp_plan(extra)

        documents = (
            '{"schema_version":"1.0.0","schema_version":"2.0.0"}',
            '{"schema_version":NaN}',
            '{"schema_version":Infinity}',
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, document in enumerate(documents):
                with self.subTest(document=document):
                    path = Path(directory) / f"invalid-{index}.json"
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(HACCPPlanError):
                        load_haccp_plan(path)


class HACCPRuntimeTests(unittest.TestCase):
    def test_all_five_limits_pass_but_only_enable_manual_review(self) -> None:
        ledger = EvidenceLedger()
        result = ledger.evaluate(good_row(), VALID_CALIBRATIONS, lot_id="LOT-PASS")
        self.assertEqual(result["overall_status"], "PASS")
        self.assertEqual(result["status_counts"], {"PASS": 5})
        self.assertEqual(result["lot_status"], "ELIGIBLE_FOR_MANUAL_REVIEW")
        self.assertTrue(result["manual_release_required"])
        self.assertFalse(result["automatic_release_performed"])
        monitors = records_of(result, "MONITORING_EVIDENCE")
        self.assertEqual(len(monitors), 5)
        self.assertEqual({item["evidence_status"] for item in monitors}, {"PASS"})
        self.assertFalse(records_of(result, "DEVIATION_RECORD"))
        self.assertFalse(records_of(result, "CONTAINMENT_ACTION"))
        self.assertEqual(len(records_of(result, "EVALUATION_SUMMARY")), 1)
        self.assertTrue(ledger.verify()["valid"])

    def test_each_runtime_limit_records_deviation_and_three_containment_actions(self) -> None:
        cases = {
            "CL-TEMP-MIN": {"safety_temp_sensor_c": 72.0},
            "CL-HOLD-MIN": {"estimated_fastest_residence_time_s": 14.9},
            "CL-FLOW-MAX": {"measured_flow_l_h": 20_500.0},
            "CL-DP-MIN": {"measured_differential_pressure_bar": 0.4},
            "CL-CIP-RELEASE": {"cip_release_permissive": 0},
        }
        for limit_id, changes in cases.items():
            with self.subTest(limit_id=limit_id):
                ledger = EvidenceLedger()
                result = ledger.evaluate(
                    good_row(**changes), VALID_CALIBRATIONS, lot_id=f"LOT-{limit_id}"
                )
                self.assertEqual(result["overall_status"], "DEVIATION")
                self.assertEqual(result["lot_status"], "HOLD")
                deviations = records_of(result, "DEVIATION_RECORD")
                self.assertEqual(len(deviations), 1)
                self.assertEqual(deviations[0]["limit_id"], limit_id)
                actions = [
                    item
                    for item in records_of(result, "CONTAINMENT_ACTION")
                    if item["limit_id"] == limit_id
                ]
                self.assertEqual(
                    {item["action"] for item in actions},
                    {"HOLD", "DIVERT", "QUARANTINE"},
                )
                self.assertTrue(
                    all(
                        item["action_status"]
                        == "RECORDED_REFERENCE_ACTION_NOT_PHYSICALLY_EXECUTED"
                        for item in actions
                    )
                )
                self.assertNotIn(
                    "RELEASE", {item.get("record_type") for item in result["records_appended"]}
                )
                self.assertTrue(ledger.verify()["valid"])

    def test_missing_and_nonfinite_measurements_are_unknown_and_never_pass(self) -> None:
        cases = {
            "missing": ("safety_temp_sensor_c", None),
            "nan": ("measured_differential_pressure_bar", math.nan),
            "infinity": ("measured_flow_l_h", math.inf),
            "missing limit reference": ("maximum_safe_flow_l_h", None),
        }
        for label, (signal, replacement) in cases.items():
            with self.subTest(label=label):
                row = good_row()
                if replacement is None:
                    del row[signal]
                else:
                    row[signal] = replacement
                ledger = EvidenceLedger()
                result = ledger.evaluate(row, VALID_CALIBRATIONS, lot_id="LOT-BAD-DATA")
                self.assertEqual(result["overall_status"], "UNKNOWN")
                self.assertEqual(result["lot_status"], "HOLD")
                relevant = [
                    item
                    for item in records_of(result, "MONITORING_EVIDENCE")
                    if item["evidence_status"] == "UNKNOWN"
                ]
                self.assertEqual(len(relevant), 1)
                self.assertIsNone(relevant[0]["observed_value"] if signal != "maximum_safe_flow_l_h" else relevant[0]["limit_value"])
                json.dumps(ledger.export_json_safe(), allow_nan=False)
                self.assertTrue(ledger.verify()["valid"])

    def test_missing_unknown_and_expired_calibration_cannot_pass(self) -> None:
        missing_ledger = EvidenceLedger()
        missing = missing_ledger.evaluate(good_row(), None, lot_id="LOT-NO-CAL")
        self.assertEqual(missing["overall_status"], "UNKNOWN")
        self.assertEqual(missing["status_counts"], {"UNKNOWN": 5})
        self.assertTrue(
            all(
                item["calibration_status"] == "UNKNOWN"
                for item in records_of(missing, "MONITORING_EVIDENCE")
            )
        )

        incomplete = dict(VALID_CALIBRATIONS)
        del incomplete["TT-104"]
        incomplete_result = EvidenceLedger().evaluate(
            good_row(), incomplete, lot_id="LOT-MISSING-CAL"
        )
        temperature = next(
            item
            for item in records_of(incomplete_result, "MONITORING_EVIDENCE")
            if item["limit_id"] == "CL-TEMP-MIN"
        )
        self.assertEqual(temperature["evidence_status"], "UNKNOWN")
        self.assertEqual(temperature["calibration_status"], "UNKNOWN")

        expired = dict(VALID_CALIBRATIONS)
        expired["FT-101"] = "EXPIRED"
        expired_result = EvidenceLedger().evaluate(
            good_row(), expired, lot_id="LOT-EXPIRED-CAL"
        )
        expired_limits = {
            item["limit_id"]
            for item in records_of(expired_result, "MONITORING_EVIDENCE")
            if item["evidence_status"] == "UNKNOWN"
        }
        self.assertEqual(expired_result["overall_status"], "UNKNOWN")
        self.assertEqual(expired_result["lot_status"], "HOLD")
        self.assertEqual(expired_limits, {"CL-HOLD-MIN", "CL-FLOW-MAX"})

        timed = dict(VALID_CALIBRATIONS)
        timed["TT-104"] = {"status": "VALID", "valid_until_time_s": 0.25}
        timed_result = EvidenceLedger().evaluate(
            good_row(time_s=0.5), timed, lot_id="LOT-TIMED-CAL"
        )
        timed_temperature = next(
            item
            for item in records_of(timed_result, "MONITORING_EVIDENCE")
            if item["limit_id"] == "CL-TEMP-MIN"
        )
        self.assertEqual(timed_temperature["evidence_status"], "UNKNOWN")
        self.assertEqual(timed_temperature["calibration_status"], "EXPIRED")

        due_now = dict(VALID_CALIBRATIONS)
        due_now["TT-104"] = {"status": "VALID", "valid_until_time_s": 0.5}
        due_result = EvidenceLedger().evaluate(
            good_row(time_s=0.5), due_now, lot_id="LOT-DUE-NOW"
        )
        due_temperature = next(
            item
            for item in records_of(due_result, "MONITORING_EVIDENCE")
            if item["limit_id"] == "CL-TEMP-MIN"
        )
        self.assertEqual(due_temperature["evidence_status"], "UNKNOWN")
        self.assertEqual(due_temperature["calibration_status"], "EXPIRED")

    def test_nonfinite_calibration_expiry_is_unknown_and_json_safe(self) -> None:
        statuses = dict(VALID_CALIBRATIONS)
        statuses["TT-104"] = {"status": "VALID", "valid_until_time_s": math.nan}
        ledger = EvidenceLedger()
        result = ledger.evaluate(good_row(), statuses, lot_id="LOT-NAN-CAL")
        temperature = next(
            item
            for item in records_of(result, "MONITORING_EVIDENCE")
            if item["limit_id"] == "CL-TEMP-MIN"
        )
        self.assertEqual(temperature["evidence_status"], "UNKNOWN")
        self.assertEqual(temperature["calibration_status"], "UNKNOWN")
        self.assertNotIn("valid_until_time_s", temperature["calibration_details"][0])
        json.dumps(ledger.export_json_safe(), allow_nan=False)

    def test_numeric_sensor_boole_and_malformed_calibration_are_unknown(self) -> None:
        for signal in (
            "safety_temp_sensor_c",
            "estimated_fastest_residence_time_s",
            "measured_flow_l_h",
            "measured_differential_pressure_bar",
        ):
            with self.subTest(signal=signal):
                result = EvidenceLedger().evaluate(
                    good_row(**{signal: True}),
                    VALID_CALIBRATIONS,
                    lot_id=f"LOT-BOOL-{signal}",
                )
                relevant = next(
                    item
                    for item in records_of(result, "MONITORING_EVIDENCE")
                    if item["signal"] == signal
                )
                self.assertEqual(relevant["evidence_status"], "UNKNOWN")
                self.assertEqual(result["lot_status"], "HOLD")

        malformed = dict(VALID_CALIBRATIONS)
        malformed["TT-104"] = {"status": "VALID", "unexpected": True}
        result = EvidenceLedger().evaluate(
            good_row(), malformed, lot_id="LOT-MALFORMED-CAL"
        )
        temperature = next(
            item
            for item in records_of(result, "MONITORING_EVIDENCE")
            if item["limit_id"] == "CL-TEMP-MIN"
        )
        self.assertEqual(temperature["evidence_status"], "UNKNOWN")
        self.assertEqual(temperature["calibration_status"], "UNKNOWN")

    def test_recovery_does_not_auto_release_a_held_lot(self) -> None:
        ledger = EvidenceLedger()
        failed = ledger.evaluate(
            good_row(time_s=0.5, safety_temp_sensor_c=71.0),
            VALID_CALIBRATIONS,
            lot_id="LOT-RECOVERY",
        )
        recovered = ledger.evaluate(
            good_row(time_s=1.0), VALID_CALIBRATIONS, lot_id="LOT-RECOVERY"
        )
        self.assertEqual(failed["overall_status"], "DEVIATION")
        self.assertEqual(recovered["overall_status"], "PASS")
        self.assertEqual(recovered["lot_status"], "HOLD")
        self.assertEqual(ledger.lot_statuses["LOT-RECOVERY"], "HOLD")
        for record in ledger.records:
            self.assertNotEqual(record.get("record_type"), "RELEASE")
            if "automatic_release_performed" in record:
                self.assertFalse(record["automatic_release_performed"])

    def test_cip_rows_are_not_misrepresented_as_production_passes(self) -> None:
        ledger = EvidenceLedger()
        result = ledger.evaluate(
            good_row(cip_cycle_active=1), VALID_CALIBRATIONS, lot_id="LOT-CIP"
        )
        self.assertEqual(result["overall_status"], "NOT_APPLICABLE")
        self.assertEqual(result["status_counts"], {"NOT_APPLICABLE": 5})
        self.assertEqual(result["lot_status"], "UNREVIEWED")
        self.assertNotIn("PASS", result["status_counts"])
        self.assertEqual(len(ledger.records), 12)

    def test_simulator_rows_can_be_evaluated_without_mutation_or_actuation(self) -> None:
        simulator = HTSTSimulator(
            HTSTConfig(duration_s=1.0, dt_s=0.5, random_seed=321)
        )
        row = simulator.run("normal")[0]
        original = copy.deepcopy(row)
        ledger = EvidenceLedger()
        result = ledger.evaluate(row, VALID_CALIBRATIONS, lot_id="LOT-SIMULATOR")
        self.assertEqual(row, original)
        self.assertIn(result["overall_status"], {"PASS", "DEVIATION", "UNKNOWN"})
        self.assertEqual(len(records_of(result, "MONITORING_EVIDENCE")), 5)
        self.assertTrue(result["shadow_only"])

    def test_invalid_rows_and_lot_identifiers_are_rejected(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "row must be a mapping"):
            EvidenceLedger().evaluate([])  # type: ignore[arg-type]
        with self.assertRaisesRegex(EvidenceError, "campaign_time_s or time_s"):
            EvidenceLedger().evaluate({}, VALID_CALIBRATIONS)
        with self.assertRaisesRegex(EvidenceError, "finite and non-negative"):
            EvidenceLedger().evaluate(good_row(time_s=math.nan), VALID_CALIBRATIONS)
        with self.assertRaisesRegex(EvidenceError, "lot_id"):
            EvidenceLedger().evaluate(good_row(), VALID_CALIBRATIONS, lot_id="bad lot")


class EvidenceIntegrityTests(unittest.TestCase):
    def test_runtime_emits_declared_calibration_and_chain_records(self) -> None:
        ledger = EvidenceLedger()
        result = ledger.evaluate(good_row(), VALID_CALIBRATIONS, lot_id="LOT-TYPES")
        calibration_records = records_of(
            result, "CALIBRATION_STATUS_REFERENCE"
        )
        monitoring_records = records_of(result, "MONITORING_EVIDENCE")
        verification_records = records_of(result, "HASH_CHAIN_VERIFICATION")
        self.assertEqual(len(calibration_records), 5)
        self.assertEqual(len(monitoring_records), 5)
        self.assertEqual(len(verification_records), 1)
        calibration_ids = {item["record_id"] for item in calibration_records}
        self.assertTrue(
            all(
                item["calibration_reference_record_id"] in calibration_ids
                for item in monitoring_records
            )
        )
        self.assertEqual(
            result["chain_verification_record_id"],
            verification_records[0]["record_id"],
        )

    def test_hash_chain_is_deterministic_and_detects_payload_or_link_tampering(self) -> None:
        first = EvidenceLedger()
        second = EvidenceLedger()
        for ledger in (first, second):
            ledger.evaluate(good_row(time_s=0.5), VALID_CALIBRATIONS, "LOT-HASH")
            ledger.evaluate(
                good_row(time_s=1.0, measured_differential_pressure_bar=0.4),
                VALID_CALIBRATIONS,
                "LOT-HASH",
            )
        self.assertEqual(first.export_json_safe(), second.export_json_safe())
        self.assertEqual(first.records[0]["previous_hash"], GENESIS_HASH)
        self.assertTrue(verify_evidence_chain(first.records)["valid"])

        payload_tamper = list(copy.deepcopy(first.records))
        payload_tamper[0]["reason"] = "tampered"
        payload_report = verify_evidence_chain(payload_tamper)
        self.assertFalse(payload_report["valid"])
        self.assertIn("hash_mismatch", payload_report["error"])

        link_tamper = list(copy.deepcopy(first.records))
        link_tamper[1]["previous_hash"] = "f" * 64
        link_report = verify_evidence_chain(link_tamper)
        self.assertFalse(link_report["valid"])
        self.assertIn("previous_hash_mismatch", link_report["error"])

        sequence_tamper = list(copy.deepcopy(first.records))
        sequence_tamper[0]["sequence"] = 2
        sequence_report = verify_evidence_chain(sequence_tamper)
        self.assertFalse(sequence_report["valid"])
        self.assertIn("sequence_mismatch", sequence_report["error"])

        anchored = verify_evidence_chain(
            first.records,
            expected_record_count=len(first.records),
            expected_chain_head=first.chain_head,
        )
        self.assertTrue(anchored["valid"])
        truncated = verify_evidence_chain(
            first.records[:-1],
            expected_record_count=len(first.records),
            expected_chain_head=first.chain_head,
        )
        self.assertFalse(truncated["valid"])
        self.assertIn("external_record_count_mismatch", truncated["error"])

    def test_records_property_is_detached_and_exports_are_json_csv_safe(self) -> None:
        ledger = EvidenceLedger()
        ledger.evaluate(good_row(), VALID_CALIBRATIONS, lot_id="LOT-EXPORT")
        detached = list(ledger.records)
        monitoring_index = next(
            index
            for index, record in enumerate(detached)
            if record["record_type"] == "MONITORING_EVIDENCE"
        )
        detached[monitoring_index]["evidence_status"] = "TAMPERED"
        self.assertTrue(ledger.verify()["valid"])
        self.assertNotEqual(
            ledger.records[monitoring_index]["evidence_status"], "TAMPERED"
        )

        detached_plan = ledger.plan
        detached_plan["controls"][0]["critical_limits"][0]["value"] = -100.0
        self.assertNotEqual(
            ledger.plan["controls"][0]["critical_limits"][0]["value"],
            -100.0,
        )

        json_artifact = ledger.export_json_safe()
        encoded = json.dumps(
            json_artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertTrue(encoded)
        self.assertEqual(json_artifact["conformity_status"], "not_assessed")
        self.assertFalse(json_artifact["certification_claim"])
        self.assertFalse(json_artifact["electronic_signature_claim"])
        self.assertFalse(json_artifact["automatic_product_release"])

        csv_artifact = ledger.export_csv_safe()
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=csv_artifact["fieldnames"])
        writer.writeheader()
        writer.writerows(csv_artifact["rows"])
        output = stream.getvalue()
        self.assertIn("record_hash", output)
        self.assertEqual(len(csv_artifact["rows"]), len(ledger.records))

    def test_evidence_contains_no_wall_clock_or_signature_claim(self) -> None:
        ledger = EvidenceLedger()
        ledger.evaluate(good_row(), VALID_CALIBRATIONS, lot_id="LOT-TIME")
        forbidden = {
            "timestamp",
            "wall_time",
            "created_at",
            "signed_at",
            "electronic_signature",
        }
        for record in ledger.records:
            self.assertFalse(forbidden.intersection(record))
            self.assertEqual(record["conformity_status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
