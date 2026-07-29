#!/usr/bin/env python3
"""Regression tests for the dependency-light FlowTwin benchmark metrics."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from flowtwin_guard.metrics import (  # noqa: E402
    classification_metrics,
    conformal_metrics,
    event_detection_metrics,
    false_alarm_metrics,
    ood_detection_metrics,
    risk_coverage_curve,
    volume_metrics,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_f1_confusion_and_absent_fixed_label(self) -> None:
        result = classification_metrics(
            ["A", "A", "B", "B"],
            ["A", "B", "B", "B"],
            labels=["A", "B", "C"],
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2.0 / 3.0 + 0.8) / 2.0)
        self.assertIsNone(result["per_class"][2]["f1"])

    def test_empty_and_invalid_classification_inputs(self) -> None:
        empty = classification_metrics([], [], labels=["A"])
        self.assertEqual(empty["rows"], 0)
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["macro_f1"])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            classification_metrics([0], [])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            classification_metrics([0], [0], labels=[0, 0])
        with self.assertRaisesRegex(ValueError, "outside labels"):
            classification_metrics([1], [1], labels=[0])


class EventMetricTests(unittest.TestCase):
    def test_one_to_one_event_metrics_use_observable_effect(self) -> None:
        time_s = np.arange(12, dtype=np.float64)
        truth = np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        # Three alarm intervals: [1,3), [8,9), [10,11).  The first was already
        # active at effect=2.5, the second detects one second after effect=7.
        alarm = np.asarray([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        effects = np.asarray(
            [np.nan, np.nan, 2.5, 2.5, 2.5, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan, np.nan]
        )
        result = event_detection_metrics(
            truth,
            alarm,
            time_s,
            np.ones(12),
            effect_time_s=effects,
            episode_ids=["episode"] * 12,
        )
        self.assertEqual(result["target_events"], 2)
        self.assertEqual(result["predicted_events"], 3)
        self.assertEqual(result["true_positive_events"], 2)
        self.assertEqual(result["false_positive_events"], 1)
        self.assertEqual(result["false_negative_events"], 0)
        self.assertAlmostEqual(result["event_precision"], 2.0 / 3.0)
        self.assertEqual(result["event_recall"], 1.0)
        self.assertAlmostEqual(result["event_f1"], 0.8)
        self.assertEqual(result["detection_latencies_s"], [0.0, 1.0])
        self.assertAlmostEqual(result["detection_latency_mean_s"], 0.5)
        self.assertAlmostEqual(result["detection_latency_median_s"], 0.5)
        self.assertAlmostEqual(result["detection_latency_p90_s"], 0.9)
        self.assertEqual(result["horizon_penalized_latency_values_s"], [0.0, 1.0])
        self.assertAlmostEqual(result["horizon_penalized_latency_mean_s"], 0.5)

    def test_latched_alarm_cannot_detect_two_events(self) -> None:
        result = event_detection_metrics(
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1],
            effect_time_s=[np.nan, 1.0, np.nan, 3.0, np.nan],
        )
        self.assertEqual(result["predicted_events"], 1)
        self.assertEqual(result["true_positive_events"], 1)
        self.assertEqual(result["false_negative_events"], 1)
        self.assertEqual(result["event_precision"], 1.0)
        self.assertEqual(result["event_recall"], 0.5)
        self.assertEqual(result["horizon_penalized_latency_values_s"], [0.0, 1.0])
        self.assertAlmostEqual(result["horizon_penalized_latency_mean_s"], 0.5)

    def test_event_zero_cases_are_undefined_not_perfect(self) -> None:
        empty = event_detection_metrics(
            [], [], [], [], effect_time_s=[]
        )
        self.assertIsNone(empty["event_precision"])
        self.assertIsNone(empty["event_recall"])
        self.assertIsNone(empty["event_f1"])
        self.assertIsNone(empty["horizon_penalized_latency_mean_s"])
        predicted_only = event_detection_metrics(
            [0], [1], [0.0], [1.0], effect_time_s=[np.nan]
        )
        self.assertEqual(predicted_only["event_precision"], 0.0)
        self.assertIsNone(predicted_only["event_recall"])
        self.assertEqual(predicted_only["event_f1"], 0.0)

    def test_event_validation_rejects_ambiguous_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant within"):
            event_detection_metrics(
                [1, 1],
                [0, 0],
                [0.0, 1.0],
                [1.0, 1.0],
                effect_time_s=[0.0, 0.5],
            )
        with self.assertRaisesRegex(ValueError, "may not precede"):
            event_detection_metrics(
                [1], [0], [1.0], [1.0], effect_time_s=[0.0]
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            event_detection_metrics(
                [0, 0],
                [0, 0],
                [0.0, 0.5],
                [1.0, 1.0],
                effect_time_s=[np.nan, np.nan],
            )
        with self.assertRaisesRegex(ValueError, "detection_eligible"):
            event_detection_metrics(
                [1, 1],
                [0, 0],
                [0.0, 1.0],
                [1.0, 1.0],
                effect_time_s=[0.0, 0.0],
                detection_eligible=[1, 0],
            )


class FalseAlarmAndVolumeMetricTests(unittest.TestCase):
    def test_false_alarm_counts_condition_onsets_and_duration(self) -> None:
        result = false_alarm_metrics(
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(result["negative_duration_s"], 4.0)
        self.assertEqual(result["false_alarm_onsets"], 2)
        self.assertEqual(result["false_alarm_duration_s"], 3.0)
        self.assertEqual(result["false_alarm_onsets_per_negative_hour"], 1800.0)
        self.assertEqual(result["false_alarm_duration_s_per_negative_hour"], 2700.0)
        self.assertEqual(result["false_alarm_duration_fraction"], 0.75)

    def test_false_alarm_has_explicit_no_negative_zero_case(self) -> None:
        result = false_alarm_metrics([1], [1], [0.0], [1.0])
        self.assertEqual(result["false_alarm_onsets"], 0)
        self.assertIsNone(result["false_alarm_onsets_per_negative_hour"])
        self.assertIsNone(result["false_alarm_duration_fraction"])

    def test_volume_totals_and_not_applicable_fields(self) -> None:
        result = volume_metrics([1.0, 2.0], [0.5, 0.0], [0.0, 4.0])
        self.assertEqual(result["unsafe_forward_volume_l"], 3.0)
        self.assertEqual(result["false_divert_volume_l"], 0.5)
        self.assertEqual(result["cooling_oos_volume_l"], 4.0)
        self.assertEqual(result["status"]["false_divert"], "reported")
        partial = volume_metrics([1.0, 2.0], None, None)
        self.assertEqual(partial["rows"], 2)
        self.assertIsNone(partial["false_divert_volume_l"])
        self.assertEqual(partial["status"]["false_divert"], "not_applicable")
        unavailable = volume_metrics(None, None, None)
        self.assertIsNone(unavailable["rows"])
        self.assertIsNone(unavailable["unsafe_forward_volume_l"])
        self.assertTrue(
            all(value == "not_applicable" for value in unavailable["status"].values())
        )

    def test_volume_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            volume_metrics([0.0], [-1.0], [0.0])
        with self.assertRaisesRegex(ValueError, "contain 2 values"):
            volume_metrics([0.0, 1.0], [0.0], None)


class ConformalAndRiskMetricTests(unittest.TestCase):
    def test_conformal_coverage_with_explicit_sets_and_membership_matrix(self) -> None:
        result = conformal_metrics([0, 1, 2, 1], [(0,), (0, 1), (), (1,)])
        self.assertEqual(result["miscoverage_count"], 1)
        self.assertEqual(result["empirical_coverage"], 0.75)
        self.assertEqual(result["mean_prediction_set_size"], 1.0)
        self.assertEqual(result["median_prediction_set_size"], 1.0)
        self.assertEqual(result["min_prediction_set_size"], 0)
        self.assertEqual(result["max_prediction_set_size"], 2)
        self.assertEqual(result["empty_set_fraction"], 0.25)
        self.assertEqual(result["singleton_fraction"], 0.5)

        matrix = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.int64)
        matrix_result = conformal_metrics([0, 2], matrix)
        self.assertEqual(matrix_result["empirical_coverage"], 1.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            conformal_metrics([0], [(0, 0)])

    def test_conformal_empty_population(self) -> None:
        result = conformal_metrics([], [])
        self.assertEqual(result["miscoverage_count"], 0)
        self.assertIsNone(result["empirical_coverage"])
        self.assertIsNone(result["mean_prediction_set_size"])

    def test_risk_coverage_is_tie_aware_bounded_and_json_safe(self) -> None:
        all_tied = risk_coverage_curve([0.0, 1.0], [0.5, 0.5])
        self.assertEqual(len(all_tied["points"]), 1)
        self.assertEqual(all_tied["points"][0]["coverage"], 1.0)
        self.assertEqual(all_tied["full_coverage_risk"], 0.5)
        self.assertEqual(all_tied["area_under_risk_coverage"], 0.5)

        many = risk_coverage_curve(
            np.arange(1000, dtype=np.float64) % 2,
            np.arange(1000, dtype=np.float64),
            max_points=11,
        )
        self.assertLessEqual(len(many["points"]), 11)
        self.assertEqual(many["points"][-1]["coverage"], 1.0)
        json.dumps(many, allow_nan=False)

    def test_risk_coverage_zero_and_validation(self) -> None:
        empty = risk_coverage_curve([], [])
        self.assertIsNone(empty["area_under_risk_coverage"])
        self.assertEqual(empty["points"], [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            risk_coverage_curve([-1.0], [1.0])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            risk_coverage_curve([0.0], [1.0], max_points=1)


class OODMetricTests(unittest.TestCase):
    def test_perfect_and_tied_ood_scores(self) -> None:
        perfect = ood_detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(perfect["aupr"], 1.0)
        self.assertEqual(perfect["fpr95"], 0.0)

        tied = ood_detection_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(tied["auroc"], 0.5)
        self.assertEqual(tied["aupr"], 0.5)
        self.assertEqual(tied["fpr95"], 1.0)

    def test_ood_single_class_and_empty_are_explicitly_undefined(self) -> None:
        for truth, scores in (([0, 0], [0.1, 0.2]), ([], [])):
            result = ood_detection_metrics(truth, scores)
            self.assertIsNone(result["auroc"])
            self.assertIsNone(result["aupr"])
            self.assertIsNone(result["fpr95"])
        with self.assertRaisesRegex(ValueError, "finite"):
            ood_detection_metrics([0, 1], [0.0, math.nan])


if __name__ == "__main__":
    unittest.main()
