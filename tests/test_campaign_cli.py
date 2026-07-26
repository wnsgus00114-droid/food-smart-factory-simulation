#!/usr/bin/env python3
"""Artifact-contract tests for the state-continuous campaign CLI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from provenance import sha256_file, verify_checksums  # noqa: E402
from run_campaign import TRANSITION_POLICY, _validated_spec, run_from_spec  # noqa: E402


class CampaignArtifactTests(unittest.TestCase):
    def test_campaign_artifacts_are_complete_and_path_independent(self) -> None:
        spec = {
            "campaign_id": "artifact-smoke",
            "config": {
                "duration_s": 90.0,
                "dt_s": 1.0,
                "random_seed": 20260725,
                "initial_fouling_index": 0.65,
            },
            "phases": [
                {"phase_id": "production", "scenario": "normal", "duration_s": 10.0},
                {"phase_id": "cip", "scenario": "cip_cycle", "duration_s": 70.0},
                {"phase_id": "restart", "scenario": "normal", "duration_s": 80.0},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            first = run_from_spec(spec_path, root / "first")
            second = run_from_spec(spec_path, root / "second")
            replay = run_from_spec(
                first / "campaign_spec.json",
                root / "replay",
                first / "initial_state.json",
            )

            first_records = verify_checksums(first, first / "checksums.sha256")
            second_records = verify_checksums(second, second / "checksums.sha256")
            replay_records = verify_checksums(replay, replay / "checksums.sha256")
            self.assertEqual(set(first_records), set(second_records))
            self.assertEqual(set(first_records), set(replay_records))
            for relative in first_records:
                self.assertEqual(
                    sha256_file(first / relative),
                    sha256_file(second / relative),
                    relative,
                )
                self.assertEqual(
                    sha256_file(first / relative),
                    sha256_file(replay / relative),
                    f"canonical artifact replay: {relative}",
                )

            manifest = json.loads((first / "run_manifest.json").read_text())
            summary = json.loads((first / "campaign_summary.json").read_text())
            transitions = json.loads((first / "transition_log.json").read_text())
            schema = json.loads((first / "schema.json").read_text())
            self.assertEqual(summary["phase_count"], 3)
            self.assertEqual(summary["row_count"], 160)
            self.assertEqual(summary["duration_s"], 160.0)
            self.assertIn("quality_out_of_spec_l", summary["totals"])
            self.assertLess(
                abs(summary["totals"]["external_volume_balance_error_l"]),
                1.0e-9,
            )
            self.assertLess(
                summary["totals"]["max_abs_mass_balance_error_l"],
                1.0e-9,
            )
            self.assertLess(
                summary["totals"]["max_abs_post_fdv_balance_error_l"],
                1.0e-9,
            )
            self.assertEqual(summary["campaign_time_start_s"], 0.0)
            self.assertEqual(summary["campaign_time_end_s"], 160.0)
            self.assertEqual(
                summary["phases"][0]["initial_fouling_index"],
                json.loads((first / "initial_state.json").read_text())["scalars"][
                    "fouling_index"
                ],
            )
            self.assertEqual(len(transitions), 3)
            self.assertTrue(all(item["previous_state_matches"] for item in transitions))
            self.assertEqual(
                manifest["initial_state_sha256"], summary["initial_state_sha256"]
            )
            self.assertEqual(
                manifest["final_state_sha256"], summary["final_state_sha256"]
            )
            self.assertIn("<artifact_dir>/campaign_spec.json", manifest["reproduce"]["command_template"])
            self.assertGreater(schema["field_count"], 202)

    def test_transition_policy_must_match_implemented_behavior(self) -> None:
        base = {
            "campaign_id": "policy-smoke",
            "config": {"duration_s": 10.0, "dt_s": 1.0},
            "phases": [
                {"phase_id": "production", "scenario": "normal", "duration_s": 1.0}
            ],
            "transition_policy": dict(TRANSITION_POLICY),
        }
        canonical = _validated_spec(base)
        self.assertEqual(canonical["transition_policy"], TRANSITION_POLICY)
        self.assertEqual(canonical["campaign_duration_s"], 1.0)
        self.assertEqual(_validated_spec(canonical), canonical)

        invalid = dict(base)
        invalid["transition_policy"] = {
            **TRANSITION_POLICY,
            "pi_integral": "retain",
        }
        with self.assertRaisesRegex(ValueError, "unsupported behavior"):
            _validated_spec(invalid)


if __name__ == "__main__":
    unittest.main()
