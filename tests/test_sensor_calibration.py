#!/usr/bin/env python3
"""Contract tests for the synthetic calibration and sensor-chain module."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from sensor_calibration import (  # noqa: E402
    CATALOG_PATH,
    SYNTHETIC_STATUS,
    CalibrationPoint,
    CalibrationRecord,
    SensorChain,
    SensorValidationError,
    combined_standard_uncertainty,
    correct_with_uncertainty,
    create_calibration_record,
    evaluate_guard_band,
    evaluate_guarded_reading,
    expanded_uncertainty,
    fit_ols,
    load_sensor_catalog,
    strict_json_loads,
)


CALIBRATED_AT = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def line_points(
    references: tuple[float, ...] = (0.0, 60.0, 120.0),
    *,
    intercept: float = 0.2,
    slope: float = 1.001,
) -> list[CalibrationPoint]:
    return [
        CalibrationPoint(reference, intercept + slope * reference)
        for reference in references
    ]


class StrictCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_sensor_catalog()

    def test_catalog_is_explicitly_synthetic_with_complete_provenance(self) -> None:
        self.assertEqual(self.catalog.schema_version, "1.0.0")
        self.assertEqual(self.catalog.catalog_status, SYNTHETIC_STATUS)
        self.assertIn("not calibration certificates", self.catalog.claim_boundary)
        self.assertEqual(
            {sensor.tag for sensor in self.catalog.sensors},
            {"TT-104", "FT-101", "PDT-101", "AIT-201", "AIT-202", "AIT-203"},
        )
        self.assertTrue(self.catalog.sources)
        self.assertTrue(
            all(source["status"] == SYNTHETIC_STATUS for source in self.catalog.sources)
        )
        for sensor in self.catalog.sensors:
            with self.subTest(tag=sensor.tag):
                self.assertEqual(sensor.status, SYNTHETIC_STATUS)
                self.assertEqual(len(sensor.parameter_provenance), 16)
                self.assertTrue(
                    all(
                        provenance.status == SYNTHETIC_STATUS
                        for _, provenance in sensor.parameter_provenance
                    )
                )

    def test_sensor_lookup_is_explicit(self) -> None:
        self.assertEqual(
            self.catalog.sensor("TT-104").unit,
            "degC",
        )
        with self.assertRaises(KeyError):
            self.catalog.sensor("UNKNOWN")

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_tokens(self) -> None:
        invalid_documents = (
            '{"tag":"A","tag":"B"}',
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(SensorValidationError):
                    strict_json_loads(document)

    def test_catalog_rejects_duplicate_tags_unknown_fields_and_nan(self) -> None:
        original = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        variants: list[tuple[str, dict[str, object]]] = []

        duplicate_tag = copy.deepcopy(original)
        duplicate_tag["sensors"][1]["tag"] = duplicate_tag["sensors"][0]["tag"]
        variants.append(("duplicate tag", duplicate_tag))

        unknown_field = copy.deepcopy(original)
        unknown_field["unexpected"] = True
        variants.append(("unknown root field", unknown_field))

        false_status = copy.deepcopy(original)
        false_status["sensors"][0]["status"] = "traceable"
        variants.append(("non-synthetic status", false_status))

        missing_provenance = copy.deepcopy(original)
        del missing_provenance["sensors"][0]["parameter_provenance"][
            "chain.lag_tau_s"
        ]
        variants.append(("missing provenance", missing_provenance))

        nonfinite = copy.deepcopy(original)
        nonfinite["sensors"][0]["chain"]["gain"] = math.nan
        variants.append(("NaN", nonfinite))

        with tempfile.TemporaryDirectory() as directory:
            for name, value in variants:
                with self.subTest(name=name):
                    path = Path(directory) / (name.replace(" ", "_") + ".json")
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(SensorValidationError):
                        load_sensor_catalog(path)


class SensorChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = load_sensor_catalog().sensor("TT-104")

    def test_chain_order_is_lag_affine_drift_hysteresis_noise_quantize(self) -> None:
        specification = replace(
            self.specification,
            lag_tau_s=2.0,
            gain=2.0,
            offset=1.0,
            drift_per_hour=1800.0,
            hysteresis=0.5,
            noise_std=0.0,
            quantization=0.25,
            dropout_probability=0.0,
        )
        chain = SensorChain(specification, seed=7, initial_true_value=0.0)
        reading = chain.step(10.0, 2.0, force_dropout=False)

        expected_filtered = 10.0 + (0.0 - 10.0) * math.exp(-1.0)
        expected_analogue = 2.0 * expected_filtered + 1.0 + 1.0 + 0.5
        expected_quantized = math.floor(expected_analogue / 0.25 + 0.5) * 0.25
        self.assertAlmostEqual(reading.filtered_value, expected_filtered)
        self.assertAlmostEqual(reading.analogue_value, expected_analogue)
        self.assertEqual(reading.noisy_value, expected_analogue)
        self.assertEqual(reading.quantized_value, expected_quantized)
        self.assertEqual(reading.value, expected_quantized)
        self.assertEqual(reading.quality, "GOOD")

        falling = chain.step(5.0, 2.0, force_dropout=False)
        expected_falling_filter = 5.0 + (expected_filtered - 5.0) * math.exp(-1.0)
        expected_falling_analogue = 2.0 * expected_falling_filter + 1.0 + 2.0 - 0.5
        self.assertAlmostEqual(falling.analogue_value, expected_falling_analogue)

    def test_same_seed_and_inputs_are_deterministic(self) -> None:
        first = SensorChain(self.specification, seed=8842, initial_true_value=68.0)
        second = SensorChain(self.specification, seed=8842, initial_true_value=68.0)
        inputs = ((69.0, 0.2), (71.0, 0.4), (73.0, 0.1), (72.0, 1.0))
        self.assertEqual(
            [first.step(value, dt).to_dict() for value, dt in inputs],
            [second.step(value, dt).to_dict() for value, dt in inputs],
        )

    def test_state_json_roundtrip_reproduces_future_noise_and_dropout(self) -> None:
        chain = SensorChain(self.specification, seed=99, initial_true_value=65.0)
        for value in (66.0, 70.0, 72.0):
            chain.step(value, 0.5)

        state_json = chain.export_state_json()
        restored = SensorChain.from_state_json(self.specification, state_json)
        future = ((73.0, 0.25), (74.0, 0.75), (71.0, 0.5), (75.0, 1.0))
        original_rows = [chain.step(value, dt).to_dict() for value, dt in future]
        restored_rows = [restored.step(value, dt).to_dict() for value, dt in future]
        self.assertEqual(original_rows, restored_rows)
        self.assertEqual(chain.export_state_json(), restored.export_state_json())

    def test_state_import_is_strict_and_bound_to_exact_specification(self) -> None:
        chain = SensorChain(self.specification, seed=3)
        chain.step(72.0, 1.0)
        state = chain.export_state()

        extra = copy.deepcopy(state)
        extra["unexpected"] = 1
        bad_fingerprint = copy.deepcopy(state)
        bad_fingerprint["specification_sha256"] = "0" * 64
        bad_rng = copy.deepcopy(state)
        bad_rng["rng_state"] = [3, [1, 2], None]
        for name, candidate in (
            ("extra key", extra),
            ("fingerprint", bad_fingerprint),
            ("random state", bad_rng),
        ):
            with self.subTest(name=name):
                with self.assertRaises(SensorValidationError):
                    SensorChain.from_state(self.specification, candidate)

        with self.assertRaises(SensorValidationError):
            SensorChain.from_state_json(
                self.specification,
                '{"schema_version":"1","schema_version":"2"}',
            )
        with self.assertRaises(SensorValidationError):
            SensorChain.from_state_json(self.specification, '{"elapsed_s":NaN}')

    def test_dropout_holds_last_and_first_dropout_has_no_value(self) -> None:
        specification = replace(
            self.specification,
            noise_std=0.0,
            dropout_probability=0.0,
        )
        chain = SensorChain(specification, seed=1, initial_true_value=70.0)
        good = chain.step(70.0, 1.0, force_dropout=False)
        held = chain.step(80.0, 1.0, force_dropout=True)
        self.assertEqual(held.value, good.value)
        self.assertEqual(held.quality, "DROPOUT_HELD")
        self.assertTrue(held.held_last)

        empty = SensorChain(specification, seed=1)
        missing = empty.step(70.0, 1.0, force_dropout=True)
        self.assertIsNone(missing.value)
        self.assertEqual(missing.quality, "DROPOUT")
        self.assertFalse(missing.held_last)

    def test_out_of_range_and_nonfinite_inputs_are_explicit(self) -> None:
        specification = replace(
            self.specification,
            gain=1.0,
            offset=0.0,
            drift_per_hour=0.0,
            hysteresis=0.0,
            noise_std=0.0,
            quantization=0.01,
            dropout_probability=0.0,
        )
        chain = SensorChain(specification, seed=2)
        self.assertEqual(chain.step(121.0, 1.0).quality, "OUT_OF_RANGE")
        for value, dt in ((math.nan, 1.0), (72.0, 0.0), (72.0, math.inf)):
            with self.subTest(value=value, dt=dt):
                with self.assertRaises(SensorValidationError):
                    chain.step(value, dt)
        with self.assertRaises(SensorValidationError):
            SensorChain(specification, seed=True)


class CalibrationMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = load_sensor_catalog().sensor("TT-104")

    def test_ols_fits_display_equals_intercept_plus_slope_reference(self) -> None:
        fit = fit_ols(line_points(intercept=1.25, slope=1.02))
        self.assertAlmostEqual(fit.intercept, 1.25)
        self.assertAlmostEqual(fit.slope, 1.02)
        self.assertAlmostEqual(fit.inverse(1.25 + 1.02 * 73.0), 73.0)
        self.assertAlmostEqual(fit.residual_std_indicated, 0.0, places=12)
        self.assertAlmostEqual(fit.parameter_standard_uncertainty(73.0), 0.0, places=12)

    def test_ols_rejects_insufficient_duplicate_degenerate_and_negative_data(self) -> None:
        invalid = (
            [CalibrationPoint(0.0, 0.0), CalibrationPoint(1.0, 1.0)],
            [
                CalibrationPoint(0.0, 0.0),
                CalibrationPoint(0.0, 0.0),
                CalibrationPoint(1.0, 1.0),
            ],
            [
                CalibrationPoint(1.0, 0.0),
                CalibrationPoint(1.0, 1.0),
                CalibrationPoint(1.0, 2.0),
            ],
            [
                CalibrationPoint(0.0, 3.0),
                CalibrationPoint(1.0, 2.0),
                CalibrationPoint(2.0, 1.0),
            ],
        )
        for points in invalid:
            with self.subTest(points=points):
                with self.assertRaises(SensorValidationError):
                    fit_ols(points)
        with self.assertRaises(SensorValidationError):
            CalibrationPoint(math.nan, 1.0)

    def test_nist_rss_and_expanded_uncertainty(self) -> None:
        combined = combined_standard_uncertainty({"u1": 3.0, "u2": 4.0})
        self.assertEqual(combined, 5.0)
        self.assertEqual(expanded_uncertainty(combined, 2.0), 10.0)
        invalid_components = ({}, {"u": -1.0}, {"u": math.nan}, {"u": True})
        for components in invalid_components:
            with self.subTest(components=components):
                with self.assertRaises(SensorValidationError):
                    combined_standard_uncertainty(components)
        with self.assertRaises(SensorValidationError):
            expanded_uncertainty(1.0, 0.0)
        with self.assertRaises(SensorValidationError):
            combined_standard_uncertainty({1: 0.1})

    def test_record_preserves_as_found_as_left_and_due_boundary(self) -> None:
        as_found = line_points(intercept=1.0, slope=1.02)
        as_left = line_points(intercept=0.2, slope=1.001)
        record = create_calibration_record(
            self.specification,
            record_id="CAL-TT-2026-001",
            calibrated_at=CALIBRATED_AT,
            as_found_points=as_found,
            as_left_points=as_left,
            uncertainty_components={"reference": 0.05},
            coverage_factor=2.0,
        )
        self.assertEqual(record.status, SYNTHETIC_STATUS)
        self.assertEqual(record.as_found_points, tuple(as_found))
        self.assertEqual(record.as_left_points, tuple(as_left))
        self.assertAlmostEqual(record.as_found_fit.slope, 1.02)
        self.assertAlmostEqual(record.as_left_fit.slope, 1.001)
        self.assertEqual(
            record.due_at,
            (CALIBRATED_AT + timedelta(days=90)).isoformat().replace("+00:00", "Z"),
        )
        self.assertEqual(record.status_at(CALIBRATED_AT - timedelta(seconds=1)), "NOT_YET_VALID")
        self.assertEqual(record.status_at(CALIBRATED_AT), "VALID")
        due = CALIBRATED_AT + timedelta(days=90)
        self.assertEqual(record.status_at(due - timedelta(microseconds=1)), "VALID")
        self.assertEqual(record.status_at(due), "EXPIRED")
        self.assertTrue(record.is_due(due))
        self.assertEqual(CalibrationRecord.from_json(record.to_json()), record)
        self.assertIn("no unbroken calibration chain", record.claim_boundary)

    def test_record_json_rejects_unknown_nan_duplicate_and_corrupt_fit(self) -> None:
        record = create_calibration_record(
            self.specification,
            record_id="CAL-STRICT",
            calibrated_at=CALIBRATED_AT,
            as_found_points=line_points(intercept=1.0, slope=1.01),
            as_left_points=line_points(),
        )
        value = record.to_dict()
        value["unexpected"] = True
        with self.assertRaises(SensorValidationError):
            CalibrationRecord.from_dict(value)

        corrupt = record.to_dict()
        corrupt["as_left_fit"]["slope"] += 0.1
        with self.assertRaises(SensorValidationError):
            CalibrationRecord.from_dict(corrupt)
        with self.assertRaises(SensorValidationError):
            CalibrationRecord.from_json('{"record_id":"A","record_id":"B"}')
        with self.assertRaises(SensorValidationError):
            CalibrationRecord.from_json('{"coverage_factor":NaN}')


class CorrectionAndGuardBandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = load_sensor_catalog().sensor("TT-104")
        self.record = create_calibration_record(
            self.specification,
            record_id="CAL-GUARD",
            calibrated_at=CALIBRATED_AT,
            as_found_points=line_points(intercept=0.8, slope=1.01),
            as_left_points=line_points(),
            uncertainty_components={"reference": 0.05},
            coverage_factor=2.0,
        )
        self.valid_at = CALIBRATED_AT + timedelta(days=1)

    @staticmethod
    def indicated_for(corrected: float) -> float:
        return 0.2 + 1.001 * corrected

    def correction(self, corrected: float):
        return correct_with_uncertainty(
            self.specification,
            self.record,
            self.indicated_for(corrected),
            observed_at=self.valid_at,
        )

    def test_inverse_as_left_correction_and_uncertainty_budget(self) -> None:
        result = self.correction(73.0)
        self.assertEqual(result.status, "VALID")
        self.assertAlmostEqual(result.corrected, 73.0)
        self.assertAlmostEqual(result.combined_standard_uncertainty, 0.05, places=12)
        self.assertAlmostEqual(result.expanded_uncertainty, 0.1, places=12)
        self.assertAlmostEqual(result.lower_bound, 72.9, places=12)
        self.assertAlmostEqual(result.upper_bound, 73.1, places=12)
        self.assertTrue(result.in_calibration_range)
        component_names = dict(result.uncertainty_components)
        self.assertIn("ols_parameter_covariance", component_names)
        self.assertIn("ols_residual", component_names)

    def test_not_yet_valid_expired_and_engineering_range_fail_closed(self) -> None:
        cases = (
            (CALIBRATED_AT - timedelta(seconds=1), self.indicated_for(73.0), "NOT_YET_VALID"),
            (CALIBRATED_AT + timedelta(days=90), self.indicated_for(73.0), "EXPIRED"),
            (self.valid_at, -1.0, "OUT_OF_RANGE"),
            (self.valid_at, 121.0, "OUT_OF_RANGE"),
        )
        for observed_at, indicated, status in cases:
            with self.subTest(status=status, indicated=indicated):
                result = correct_with_uncertainty(
                    self.specification,
                    self.record,
                    indicated,
                    observed_at=observed_at,
                )
                self.assertEqual(result.status, status)
                self.assertIsNone(result.expanded_uncertainty)

    def test_calibration_range_blocks_extrapolation_unless_explicit(self) -> None:
        narrow = create_calibration_record(
            self.specification,
            record_id="CAL-NARROW",
            calibrated_at=CALIBRATED_AT,
            as_found_points=line_points((60.0, 72.0, 80.0)),
            as_left_points=line_points((60.0, 72.0, 80.0)),
            uncertainty_components={"reference": 0.05},
        )
        indicated = self.indicated_for(90.0)
        blocked = correct_with_uncertainty(
            self.specification,
            narrow,
            indicated,
            observed_at=self.valid_at,
        )
        self.assertEqual(blocked.status, "OUT_OF_RANGE")
        self.assertFalse(blocked.in_calibration_range)
        allowed = correct_with_uncertainty(
            self.specification,
            narrow,
            indicated,
            observed_at=self.valid_at,
            allow_extrapolation=True,
        )
        self.assertEqual(allowed.status, "VALID")
        self.assertFalse(allowed.in_calibration_range)
        self.assertIn("extrapolated", allowed.reason)

    def test_lower_guard_band_pass_fail_and_overlap_are_distinct(self) -> None:
        expected = ((72.2, "PASS"), (71.8, "FAIL"), (72.0, "UNKNOWN"))
        for corrected, status in expected:
            with self.subTest(corrected=corrected):
                decision = evaluate_guard_band(
                    self.specification,
                    self.correction(corrected),
                )
                self.assertEqual(decision.status, status)
                if status == "PASS":
                    self.assertEqual(decision.disposition, "NO_AUTOMATIC_RELEASE")
                else:
                    self.assertEqual(decision.disposition, "HOLD")

    def test_upper_guard_band_uses_upper_uncertainty_bound(self) -> None:
        cases = ((71.8, "PASS"), (72.2, "FAIL"), (72.0, "UNKNOWN"))
        for corrected, expected in cases:
            with self.subTest(corrected=corrected):
                decision = evaluate_guard_band(
                    self.specification,
                    self.correction(corrected),
                    critical_limit=72.0,
                    direction="upper",
                )
                self.assertEqual(decision.status, expected)

    def test_expired_missing_or_bad_quality_reading_is_held_unknown(self) -> None:
        expired = correct_with_uncertainty(
            self.specification,
            self.record,
            self.indicated_for(73.0),
            observed_at=CALIBRATED_AT + timedelta(days=90),
        )
        decision = evaluate_guard_band(self.specification, expired)
        self.assertEqual((decision.status, decision.disposition), ("UNKNOWN", "HOLD"))

        chain = SensorChain(
            replace(self.specification, dropout_probability=0.0),
            seed=2,
        )
        dropped = chain.step(73.0, 1.0, force_dropout=True)
        missing = evaluate_guarded_reading(
            self.specification,
            dropped,
            None,
            observed_at=self.valid_at,
        )
        self.assertEqual((missing.status, missing.disposition), ("UNKNOWN", "HOLD"))
        bad_quality = evaluate_guarded_reading(
            self.specification,
            dropped,
            self.record,
            observed_at=self.valid_at,
        )
        self.assertEqual((bad_quality.status, bad_quality.disposition), ("UNKNOWN", "HOLD"))

    def test_incomplete_or_inconsistent_valid_interval_fails_closed(self) -> None:
        valid = self.correction(73.0)
        invalid_results = (
            replace(valid, lower_bound=None),
            replace(valid, lower_bound=74.0),
            replace(valid, expanded_uncertainty=math.nan),
        )
        for result in invalid_results:
            with self.subTest(result=result):
                decision = evaluate_guard_band(self.specification, result)
                self.assertEqual((decision.status, decision.disposition), ("UNKNOWN", "HOLD"))

    def test_additional_uncertainty_is_combined_and_duplicates_rejected(self) -> None:
        result = correct_with_uncertainty(
            self.specification,
            self.record,
            self.indicated_for(73.0),
            observed_at=self.valid_at,
            additional_uncertainty_components={"environment": 0.12},
        )
        self.assertAlmostEqual(result.combined_standard_uncertainty, 0.13, places=12)
        with self.assertRaises(SensorValidationError):
            correct_with_uncertainty(
                self.specification,
                self.record,
                self.indicated_for(73.0),
                observed_at=self.valid_at,
                additional_uncertainty_components={"reference": 0.1},
            )


if __name__ == "__main__":
    unittest.main()
