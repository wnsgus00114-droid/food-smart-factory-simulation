#!/usr/bin/env python3
"""Contract tests for the independent multi-cycle lifecycle/RUL layer."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_DIR))

from lifecycle import (  # noqa: E402
    EVENT_FIELDS,
    SERVICE_INTERVAL_FIELDS,
    TRACE_FIELDS,
    PolicyValidationError,
    conditional_median_next_event_rul,
    load_maintenance_policy,
    simulate_lifecycle,
    validate_policy,
)
from provenance import verify_checksums  # noqa: E402


POLICY = MODULE_DIR / "maintenance_policy.json"
GENERATE = MODULE_DIR / "generate_multicycle_rul_dataset.py"


class LifecyclePolicyAndPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_maintenance_policy(POLICY)

    def test_policy_is_strict_and_rejects_ambiguous_or_nonfinite_inputs(self) -> None:
        self.assertEqual(self.policy.schema_version, "1.0.0")
        self.assertEqual(len(self.policy.components), 4)
        self.assertEqual(
            {item.asset_id for item in self.policy.components},
            {"HX-101", "P-101", "XV-101", "TT-102B"},
        )

        mutations: list[tuple[str, object, str]] = []
        unknown = self.policy.to_dict()
        unknown["silent_typo"] = True
        mutations.append(("unknown key", unknown, "keys differ"))

        duplicate = self.policy.to_dict()
        duplicate["components"][1]["asset_id"] = duplicate["components"][0]["asset_id"]
        mutations.append(("duplicate asset", duplicate, "duplicate component asset_id"))

        ineffective_complete = self.policy.to_dict()
        ineffective_complete["cip_recipes"]["complete"][
            "fouling_removal_fraction"
        ] = ineffective_complete["cip_recipes"]["incomplete"][
            "fouling_removal_fraction"
        ]
        mutations.append(
            ("CIP ordering", ineffective_complete, "complete CIP must remove more")
        )

        nonfinite = self.policy.to_dict()
        nonfinite["components"][0]["weibull_shape"] = float("nan")
        mutations.append(("non-finite", nonfinite, "must be finite"))

        bad_replacement = self.policy.to_dict()
        bad_replacement["corrective_action"]["virtual_age_factor"] = 0.1
        mutations.append(("non-renewing replacement", bad_replacement, "zero-factor"))

        for label, payload, message in mutations:
            with self.subTest(label=label):
                with self.assertRaisesRegex(PolicyValidationError, message):
                    validate_policy(payload)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temporary:
            duplicate_json = Path(temporary) / "duplicate.json"
            duplicate_json.write_text(
                '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyValidationError, "duplicate JSON key"):
                load_maintenance_policy(duplicate_json)

    def test_cip_is_partial_and_actions_follow_kijima_two_factors(self) -> None:
        result = simulate_lifecycle(self.policy, cycles=18, seed=123)
        self.assertEqual(len(result.trace), 4 * 19)
        cip_events = [item for item in result.events if item["event_type"] == "cip"]
        self.assertEqual(len(cip_events), 4 * 18)
        recipes = self.policy.recipe_by_name
        saw_positive_fouling = False
        for event in cip_events:
            before_fouling = float(event["fouling_before"])
            after_fouling = float(event["fouling_after"])
            before_damage = float(event["damage_before"])
            after_damage = float(event["damage_after"])
            recipe = recipes[str(event["cip_kind"])]
            self.assertAlmostEqual(
                after_fouling,
                before_fouling * (1.0 - recipe.fouling_removal_fraction),
                places=12,
            )
            self.assertGreaterEqual(after_damage + 1.0e-15, before_damage)
            self.assertEqual(
                float(event["virtual_age_before_h"]),
                float(event["virtual_age_after_h"]),
            )
            saw_positive_fouling |= before_fouling > 0.0
        self.assertTrue(saw_positive_fouling)

        maintenance = {
            (str(event["asset_id"]), int(event["cycle"])): event
            for event in result.events
            if event["event_type"] == "scheduled_maintenance"
        }
        expected = {
            ("TT-102B", 6): ("calibration", 1.0, 0.35, 1.0),
            ("P-101", 12): ("minor_repair", 0.65, 0.60, 1.0),
            ("XV-101", 15): ("minor_repair", 0.70, 0.65, 1.0),
            ("HX-101", 18): ("overhaul", 0.25, 0.25, 0.05),
        }
        for key, (name, q, damage_factor, fouling_factor) in expected.items():
            with self.subTest(asset_cycle=key):
                event = maintenance[key]
                self.assertEqual(event["maintenance_action"], name)
                self.assertAlmostEqual(float(event["virtual_age_factor_q"]), q)
                self.assertAlmostEqual(
                    float(event["virtual_age_after_h"]),
                    q * float(event["virtual_age_before_h"]),
                )
                self.assertAlmostEqual(
                    float(event["damage_after"]),
                    damage_factor * float(event["damage_before"]),
                )
                self.assertAlmostEqual(
                    float(event["fouling_after"]),
                    fouling_factor * float(event["fouling_before"]),
                )
        self.assertTrue(result.summary["all_last_intervals_right_censored"])

    def test_conditional_median_rul_matches_closed_form_competing_risks(self) -> None:
        hx = self.policy.component_by_id["HX-101"]
        value = conditional_median_next_event_rul(
            hx,
            virtual_age_h=0.0,
            irreversible_damage=0.0,
            reversible_fouling=0.0,
            load_factor=1.0,
        )
        expected_weibull = hx.weibull_scale_h * math.log(2.0) ** (
            1.0 / hx.weibull_shape
        )
        expected_damage = hx.damage_limit / hx.damage_rate_per_operating_h
        expected_fouling = hx.fouling_limit / hx.fouling_rate_per_operating_h
        self.assertAlmostEqual(float(value["weibull_median_rul_h"]), expected_weibull)
        self.assertAlmostEqual(float(value["damage_limit_rul_h"]), expected_damage)
        self.assertAlmostEqual(float(value["fouling_limit_rul_h"]), expected_fouling)
        self.assertEqual(value["median_competing_cause"], "FOULING_LIMIT")
        self.assertAlmostEqual(
            float(value["conditional_median_next_event_rul_h"]), expected_fouling
        )

        aged = conditional_median_next_event_rul(
            hx,
            virtual_age_h=100.0,
            irreversible_damage=0.2,
            reversible_fouling=0.4,
            load_factor=2.0,
        )
        self.assertGreaterEqual(float(aged["conditional_median_next_event_rul_h"]), 0.0)
        self.assertLess(float(aged["fouling_limit_rul_h"]), expected_fouling)

    def _accelerated_policy(
        self,
        *,
        damage_rate: float,
        fouling_rate: float,
        weibull_scale_h: float,
        production_hours: float,
    ):
        raw = self.policy.to_dict()
        component = copy.deepcopy(raw["components"][0])
        component["weibull_shape"] = 1.0
        component["weibull_scale_h"] = weibull_scale_h
        component["damage_rate_per_operating_h"] = damage_rate
        component["cip_damage_per_exposure"] = 0.0
        component["fouling_rate_per_operating_h"] = fouling_rate
        raw["components"] = [component]
        raw["scheduled_actions"] = []
        raw["production_hours_per_cycle"] = production_hours
        raw["default_cycles"] = 2
        return validate_policy(raw)

    def test_all_competing_causes_recur_and_last_interval_is_right_censored(self) -> None:
        cases = {
            "FOULING_LIMIT": self._accelerated_policy(
                damage_rate=0.05,
                fouling_rate=0.20,
                weibull_scale_h=1.0e12,
                production_hours=6.0,
            ),
            "DAMAGE_LIMIT": self._accelerated_policy(
                damage_rate=0.25,
                fouling_rate=0.05,
                weibull_scale_h=1.0e12,
                production_hours=6.0,
            ),
            "WEIBULL_FAILURE": self._accelerated_policy(
                damage_rate=0.0,
                fouling_rate=0.0,
                weibull_scale_h=0.20,
                production_hours=1.0,
            ),
        }
        for cause, policy in cases.items():
            with self.subTest(cause=cause):
                first = simulate_lifecycle(policy, cycles=2, seed=123)
                second = simulate_lifecycle(policy, cycles=2, seed=123)
                self.assertEqual(first.trace, second.trace)
                self.assertEqual(first.events, second.events)
                failures = [
                    event for event in first.events if event["event_type"] == "failure"
                ]
                self.assertGreaterEqual(len(failures), 2)
                self.assertEqual({event["failure_cause"] for event in failures}, {cause})
                self.assertTrue(
                    all(event["maintenance_action"] == "replacement" for event in failures)
                )
                self.assertTrue(
                    all(float(event["virtual_age_after_h"]) == 0.0 for event in failures)
                )
                observed = [
                    interval
                    for interval in first.service_intervals
                    if interval["event_observed"] == 1
                ]
                self.assertEqual(len(observed), len(failures))
                self.assertTrue(
                    all(
                        interval["exact_rul_h"] is not None
                        and interval["rul_lower_bound_h"] is None
                        for interval in observed
                    )
                )
                last = [
                    interval
                    for interval in first.service_intervals
                    if interval["is_last_interval"] == 1
                ]
                self.assertEqual(len(last), 1)
                self.assertEqual(last[0]["terminal_event_type"], "administrative_censor")
                self.assertEqual(last[0]["event_observed"], 0)
                self.assertIsNone(last[0]["exact_rul_h"])
                self.assertIsNotNone(last[0]["rul_lower_bound_h"])


class LifecycleCLIContractTests(unittest.TestCase):
    @staticmethod
    def _run(command: list[str], *, allow_failure: bool = False):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
        if process.returncode and not allow_failure:
            raise AssertionError(
                f"command failed ({process.returncode}): {' '.join(command)}\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        return process

    @staticmethod
    def _csv_header(path: Path) -> list[str]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader)

    def test_cli_is_byte_reproducible_path_independent_atomic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="htst-lifecycle-cli-") as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            base = [
                sys.executable,
                str(GENERATE),
                "--policy",
                str(POLICY),
                "--cycles",
                "18",
                "--seed",
                "456",
            ]
            self._run([*base, "--output", str(first)])
            self._run([*base, "--output", str(second)])
            expected = {
                "trace.csv",
                "events.csv",
                "service_intervals.csv",
                "summary.json",
                "schema.json",
                "run_manifest.json",
                "checksums.sha256",
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected)
            self.assertEqual({path.name for path in second.iterdir()}, expected)
            for name in expected:
                with self.subTest(reproducible=name):
                    self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            verified = verify_checksums(first, first / "checksums.sha256")
            self.assertEqual(set(verified), expected - {"checksums.sha256"})
            for relative, digest in verified.items():
                self.assertEqual(
                    hashlib.sha256((first / relative).read_bytes()).hexdigest(), digest
                )
            self.assertEqual(self._csv_header(first / "trace.csv"), list(TRACE_FIELDS))
            self.assertEqual(self._csv_header(first / "events.csv"), list(EVENT_FIELDS))
            self.assertEqual(
                self._csv_header(first / "service_intervals.csv"),
                list(SERVICE_INTERVAL_FIELDS),
            )
            summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (first / "run_manifest.json").read_text(encoding="utf-8")
            )
            schema = json.loads((first / "schema.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["all_last_intervals_right_censored"])
            self.assertEqual(manifest["run_id"], summary["run_id"])
            self.assertEqual(manifest["policy_hash"], summary["policy_hash"])
            self.assertNotIn(str(root), json.dumps(manifest, ensure_ascii=False))
            self.assertIn("<maintenance_policy.json>", manifest["reproduce"])
            self.assertEqual(
                schema["tables"]["trace.csv"]["field_count"], len(TRACE_FIELDS)
            )

            stale = first / "stale.txt"
            stale.write_text("remove after success\n", encoding="utf-8")
            self._run([*base, "--output", str(first)])
            self.assertFalse(stale.exists())
            for name in expected:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            invalid_payload = load_maintenance_policy(POLICY).to_dict()
            invalid_payload["unknown_key"] = "must fail before publish"
            invalid_policy = root / "invalid.json"
            invalid_policy.write_text(
                json.dumps(invalid_payload, ensure_ascii=False), encoding="utf-8"
            )
            preserved = root / "preserved"
            preserved.mkdir()
            sentinel = preserved / "existing.txt"
            sentinel.write_text("keep me\n", encoding="utf-8")
            failed = self._run(
                [
                    sys.executable,
                    str(GENERATE),
                    "--policy",
                    str(invalid_policy),
                    "--output",
                    str(preserved),
                ],
                allow_failure=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("keys differ from contract", failed.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(list(root.glob(".preserved.staging-*")))


if __name__ == "__main__":
    unittest.main()
