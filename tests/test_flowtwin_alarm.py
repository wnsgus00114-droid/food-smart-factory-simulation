#!/usr/bin/env python3
"""Regression tests for the validation-only operational alarm policy."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from flowtwin_guard.alarm import (  # noqa: E402
    ALARM_POLICY_VERSION,
    AlarmPolicyConfig,
    apply_alarm_policy,
    fit_alarm_policy,
)


class AlarmStateMachineTests(unittest.TestCase):
    def test_config_contract_and_json_round_trip(self) -> None:
        config = AlarmPolicyConfig(0.8, 0.4, 1.5, 2.0, 3.0)
        payload = config.to_dict()
        self.assertEqual(payload["version"], ALARM_POLICY_VERSION)
        self.assertEqual(AlarmPolicyConfig.from_dict(payload), config)
        json.dumps(payload, allow_nan=False)

        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            AlarmPolicyConfig(0.2, 0.3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            AlarmPolicyConfig(0.8, 0.4, min_on_duration_s=-1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            AlarmPolicyConfig(float("nan"), 0.4)
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            AlarmPolicyConfig.from_dict({**payload, "unsafe_override": True})

    def test_causal_hysteresis_persistence_cooldown_and_episode_reset(self) -> None:
        config = AlarmPolicyConfig(
            on_threshold=0.8,
            off_threshold=0.3,
            min_on_duration_s=1.5,
            min_off_duration_s=1.0,
            cooldown_s=1.5,
        )
        # Episode A asserts after 0.5 + 1.0 seconds above on, clears after
        # 0.5 + 0.5 seconds below off, and cannot immediately reassert.
        # Episode B starts from a clean state despite A ending above on.
        result = apply_alarm_policy(
            scores=[0.9, 0.9, 0.5, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9],
            step_dt_s=[0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5],
            episode_ids=["A"] * 7 + ["B"] * 2,
            config=config,
        )
        self.assertEqual(
            result.tolist(),
            [False, True, True, True, False, False, False, False, True],
        )

    def test_interleaved_episodes_have_independent_state(self) -> None:
        result = apply_alarm_policy(
            scores=[1.0, 1.0, 1.0, 0.0],
            step_dt_s=[1.0, 1.0, 1.0, 1.0],
            episode_ids=["A", "B", "A", "B"],
            config=AlarmPolicyConfig(0.8, 0.2, min_on_duration_s=2.0),
        )
        self.assertEqual(result.tolist(), [False, False, True, False])

    def test_input_validation(self) -> None:
        config = AlarmPolicyConfig(0.8, 0.2)
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_alarm_policy([0.5], [0.0], ["e"], config)
        with self.assertRaisesRegex(ValueError, "contain 1 values"):
            apply_alarm_policy([0.5], [1.0], [], config)
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_alarm_policy([float("nan")], [1.0], ["e"], config)


class AlarmPolicyFitTests(unittest.TestCase):
    @staticmethod
    def _validation_data() -> dict[str, object]:
        return {
            "scores": np.asarray([0.6, 0.1, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8]),
            "truth_active": np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
            "time_s": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=float),
            "step_dt_s": np.ones(8),
            "effect_time_s": np.asarray(
                [np.nan, np.nan, 2.0, 2.0, np.nan, np.nan, 2.0, 2.0]
            ),
            "episode_ids": ["e1"] * 4 + ["e2"] * 4,
            "profile_ids": ["p1"] * 4 + ["p2"] * 4,
            "splits": ["validation_id"] * 8,
        }

    def test_grid_search_enforces_each_profile_and_is_deterministic(self) -> None:
        data = self._validation_data()
        first = fit_alarm_policy(
            **data,
            on_thresholds=[0.9, 0.7, 0.5],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=1000.0,
            objective="event_f1",
        )
        reversed_grid = fit_alarm_policy(
            **data,
            on_thresholds=[0.5, 0.7, 0.9],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=1000.0,
            objective="event_f1",
        )
        self.assertEqual(first["status"], "selected")
        self.assertEqual(first["config"], reversed_grid["config"])
        self.assertEqual(first["config"]["on_threshold"], 0.7)
        selected = first["audit"]["selected_validation_metrics"]
        self.assertEqual(selected["event_metrics"]["event_f1"], 1.0)
        self.assertEqual(
            selected["profile_false_alarm_audit"][
                "maximum_profile_false_alarm_onsets_per_negative_hour"
            ],
            0.0,
        )
        # Threshold 0.5 is rejected because p1 has one false onset in two
        # seconds of negative exposure, even though p2 has none.
        threshold_05 = next(
            candidate
            for candidate in first["audit"]["candidates"]
            if candidate["config"]["on_threshold"] == 0.5
        )
        self.assertFalse(threshold_05["feasible"])
        self.assertEqual(
            threshold_05["profile_false_alarm_audit"][
                "maximum_profile_false_alarm_onsets_per_negative_hour"
            ],
            1800.0,
        )
        self.assertEqual(
            threshold_05["profile_false_alarm_audit"][
                "pooled_false_alarm_onsets_per_negative_hour"
            ],
            900.0,
        )
        self.assertLessEqual(
            threshold_05["profile_false_alarm_audit"][
                "pooled_false_alarm_onsets_per_negative_hour"
            ],
            first["audit"]["constraint"]["limit"],
        )
        json.dumps(first, allow_nan=False)

    def test_event_recall_objective_and_deterministic_conservative_tie(self) -> None:
        data = self._validation_data()
        result = fit_alarm_policy(
            **data,
            on_thresholds=[0.7, 0.75],
            off_thresholds=[0.2, 0.3],
            min_on_durations_s=[0.0, 0.5],
            min_off_durations_s=[0.0],
            cooldowns_s=[0.0],
            max_false_alarm_onsets_per_negative_hour=0.0,
            objective="event_recall",
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["audit"]["objective"], "event_recall")
        self.assertEqual(result["config"]["on_threshold"], 0.75)
        self.assertEqual(result["config"]["off_threshold"], 0.3)
        self.assertEqual(result["config"]["min_on_duration_s"], 0.5)

    def test_fit_fails_closed_for_non_validation_split_vocabulary(self) -> None:
        data = self._validation_data()
        for splits in (
            ["train"] * 8,
            ["validation_id"] * 7 + ["test_id"],
            [],
        ):
            contaminated = {**data, "splits": splits}
            with self.assertRaisesRegex(ValueError, "validation-only"):
                fit_alarm_policy(
                    **contaminated,
                    on_thresholds=[0.7],
                    off_thresholds=[0.3],
                    max_false_alarm_onsets_per_negative_hour=0.0,
                )

    def test_no_operating_point_is_explicit_and_json_safe(self) -> None:
        data = self._validation_data()
        result = fit_alarm_policy(
            **data,
            on_thresholds=[0.5],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=0.0,
        )
        self.assertEqual(result["status"], "no_operating_point")
        self.assertIsNone(result["config"])
        self.assertEqual(result["audit"]["feasible_candidate_count"], 0)
        self.assertIn(
            "profile_false_alarm_constraint_exceeded",
            result["audit"]["no_operating_point_reasons"],
        )
        json.dumps(result, allow_nan=False)

    def test_minimum_event_recall_rejects_a_no_alarm_policy(self) -> None:
        data = self._validation_data()
        without_floor = fit_alarm_policy(
            **data,
            on_thresholds=[0.5, 0.9],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=0.0,
        )
        self.assertEqual(without_floor["status"], "selected")
        self.assertEqual(without_floor["config"]["on_threshold"], 0.9)
        self.assertEqual(
            without_floor["audit"]["selected_validation_metrics"][
                "event_metrics"
            ]["event_recall"],
            0.0,
        )

        with_floor = fit_alarm_policy(
            **data,
            on_thresholds=[0.5, 0.9],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=0.0,
            min_event_recall=0.5,
        )
        self.assertEqual(with_floor["status"], "no_operating_point")
        self.assertIsNone(with_floor["config"])
        self.assertEqual(
            with_floor["audit"]["constraint"]["minimum_event_recall"], 0.5
        )
        self.assertEqual(
            with_floor["audit"]["constraint"][
                "minimum_event_recall_operator"
            ],
            ">=",
        )
        self.assertIn(
            "minimum_event_recall_not_met",
            with_floor["audit"]["no_operating_point_reasons"],
        )
        json.dumps(with_floor, allow_nan=False)

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            fit_alarm_policy(
                **data,
                on_thresholds=[0.9],
                off_thresholds=[0.3],
                max_false_alarm_onsets_per_negative_hour=0.0,
                min_event_recall=1.1,
            )

    def test_profile_without_negative_exposure_fails_closed(self) -> None:
        data = self._validation_data()
        data["truth_active"] = np.ones(8, dtype=int)
        data["effect_time_s"] = np.asarray([0.0] * 4 + [0.0] * 4)
        result = fit_alarm_policy(
            **data,
            on_thresholds=[0.7],
            off_thresholds=[0.3],
            max_false_alarm_onsets_per_negative_hour=10.0,
        )
        self.assertEqual(result["status"], "no_operating_point")
        self.assertIn(
            "profile_without_negative_exposure",
            result["audit"]["no_operating_point_reasons"],
        )

    def test_minimum_profile_event_recall_rejects_pooled_pass(self) -> None:
        data = {
            "scores": np.asarray([0.1, 0.9, 0.9, 0.1, 0.1, 0.4, 0.4, 0.1]),
            "truth_active": np.asarray([0, 1, 1, 0, 0, 1, 1, 0]),
            "time_s": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=float),
            "step_dt_s": np.ones(8),
            "effect_time_s": np.asarray(
                [np.nan, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0, np.nan]
            ),
            "episode_ids": ["e1"] * 4 + ["e2"] * 4,
            "profile_ids": ["p1"] * 4 + ["p2"] * 4,
            "splits": ["validation_id"] * 8,
        }
        pooled_only = fit_alarm_policy(
            **data,
            on_thresholds=[0.8],
            off_thresholds=[0.2],
            max_false_alarm_onsets_per_negative_hour=0.0,
            min_event_recall=0.5,
        )
        self.assertEqual(pooled_only["status"], "selected")
        self.assertEqual(
            pooled_only["audit"]["selected_validation_metrics"]["event_metrics"][
                "event_recall"
            ],
            0.5,
        )

        profiled = fit_alarm_policy(
            **data,
            on_thresholds=[0.8],
            off_thresholds=[0.2],
            max_false_alarm_onsets_per_negative_hour=0.0,
            min_event_recall=0.5,
            min_profile_event_recall=0.5,
        )
        self.assertEqual(profiled["status"], "no_operating_point")
        self.assertIn(
            "minimum_profile_event_recall_not_met",
            profiled["audit"]["no_operating_point_reasons"],
        )
        candidate_audit = profiled["audit"]["candidates"][0][
            "profile_event_recall_audit"
        ]
        self.assertEqual(candidate_audit["minimum_profile_event_recall"], 0.0)
        self.assertEqual(candidate_audit["rejected_profiles"], ["p2"])
        p2 = next(
            row for row in candidate_audit["profiles"] if row["profile_id"] == "p2"
        )
        self.assertEqual(p2["event_recall"], 0.0)
        self.assertEqual(
            p2["rejection_reasons"],
            ["minimum_profile_event_recall_not_met"],
        )
        json.dumps(profiled, allow_nan=False)

    def test_profile_without_eligible_event_is_explicitly_excluded(self) -> None:
        data = {
            "scores": np.asarray([0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "truth_active": np.asarray([0, 1, 1, 0, 0, 1, 1, 0]),
            "time_s": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=float),
            "step_dt_s": np.ones(8),
            "effect_time_s": np.asarray(
                [np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan]
            ),
            "episode_ids": ["e1"] * 4 + ["e2"] * 4,
            "profile_ids": ["p1"] * 4 + ["p2"] * 4,
            "splits": ["validation_id"] * 8,
            "detection_eligible": np.asarray([1, 1, 1, 1, 0, 0, 0, 0]),
        }
        result = fit_alarm_policy(
            **data,
            on_thresholds=[0.8],
            off_thresholds=[0.2],
            max_false_alarm_onsets_per_negative_hour=0.0,
            min_profile_event_recall=1.0,
        )
        self.assertEqual(result["status"], "selected")
        audit = result["audit"]["selected_validation_metrics"][
            "profile_event_recall_audit"
        ]
        self.assertEqual(audit["eligible_profile_count"], 1)
        self.assertEqual(audit["excluded_profile_count"], 1)
        self.assertEqual(audit["profiles_with_eligible_events"], ["p1"])
        self.assertEqual(audit["profiles_without_eligible_events"], ["p2"])
        self.assertTrue(audit["profiles_without_eligible_events_are_excluded"])
        p2 = next(row for row in audit["profiles"] if row["profile_id"] == "p2")
        self.assertFalse(p2["constraint_applicable"])
        self.assertIsNone(p2["constraint_satisfied"])
        self.assertEqual(
            p2["constraint_exclusion_reason"], "no_eligible_truth_events"
        )
        self.assertEqual(p2["rejection_reasons"], [])
        json.dumps(result, allow_nan=False)

    def test_minimum_profile_event_recall_validation(self) -> None:
        data = self._validation_data()
        for invalid in (-0.1, 1.1, float("nan"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_alarm_policy(
                    **data,
                    on_thresholds=[0.7],
                    off_thresholds=[0.3],
                    max_false_alarm_onsets_per_negative_hour=0.0,
                    min_profile_event_recall=invalid,
                )


if __name__ == "__main__":
    unittest.main()
