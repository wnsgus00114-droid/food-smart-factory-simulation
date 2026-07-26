#!/usr/bin/env python3
"""End-to-end contracts for the integrated non-biological digital-twin runner."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from haccp import verify_evidence_chain  # noqa: E402
from provenance import sha256_file, verify_checksums  # noqa: E402
from run_digital_twin import (  # noqa: E402
    DEFAULT_CONFIG,
    DigitalTwinConfigError,
    _load_config,
    run_integrated,
)


EXPECTED_ARTIFACTS = {
    "calibration_records.json",
    "calibration_summary.csv",
    "config_snapshot.json",
    "digital_twin_summary.json",
    "haccp_deviations.csv",
    "haccp_evidence.jsonl",
    "lifecycle_events.csv",
    "lifecycle_trace.csv",
    "plc_events.csv",
    "plc_trace.csv",
    "process_events.csv",
    "process_trace.csv",
    "reference_plant.md",
    "reference_plant_snapshot.json",
    "run_manifest.json",
    "schema.json",
    "sensor_trace.csv",
    "service_intervals.csv",
    "timeseries.csv",
}


class DigitalTwinIntegrationTests(unittest.TestCase):
    def _minimal_config(self, root: Path) -> Path:
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["process"].update(
            {
                "process_cycles": 1,
                "production_duration_s": 1.0,
                "cip_duration_s": 540.0,
                "restart_duration_s": 1.0,
                "cip_pattern": ["complete"],
            }
        )
        config["lifecycle"]["cycles"] = 1
        config_path = root / "digital_twin_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        shutil.copy2(MODULE_DIR / "maintenance_policy.json", root / "maintenance_policy.json")
        return config_path

    def test_default_config_is_explicitly_non_biological(self) -> None:
        config = _load_config(DEFAULT_CONFIG)
        self.assertNotIn("biology", config)
        self.assertNotIn("biological", config)
        self.assertNotIn("microbiology", config)
        self.assertTrue(config["plc"]["shadow_only"])
        self.assertFalse(config["haccp"]["automatic_release"])
        self.assertEqual(config["haccp"]["jurisdiction_status"], "not_assessed")

    def test_integrated_artifacts_are_reproducible_and_self_verifying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._minimal_config(root)
            first = run_integrated(config_path, root / "first")
            second = run_integrated(config_path, root / "second")

            first_records = verify_checksums(first, first / "checksums.sha256")
            second_records = verify_checksums(second, second / "checksums.sha256")
            self.assertEqual(set(first_records), EXPECTED_ARTIFACTS)
            self.assertEqual(set(first_records), set(second_records))
            for relative in first_records:
                self.assertEqual(
                    sha256_file(first / relative),
                    sha256_file(second / relative),
                    relative,
                )

            summary = json.loads(
                (first / "digital_twin_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (first / "run_manifest.json").read_text(encoding="utf-8")
            )
            schema = json.loads((first / "schema.json").read_text(encoding="utf-8"))
            evidence = [
                json.loads(line)
                for line in (first / "haccp_evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertFalse(summary["biological_model"]["implemented"])
            self.assertEqual(manifest["components"]["biological_model"], "excluded")
            self.assertEqual(summary["process"]["rows"], 542)
            self.assertEqual(summary["instrumentation"]["sensor_trace_rows"], 6 * 542)
            self.assertEqual(summary["plc_shadow"]["scan_count"], 542)
            self.assertTrue(summary["haccp_evidence"]["chain_valid"])
            chain = verify_evidence_chain(
                evidence,
                expected_record_count=summary["haccp_evidence"]["record_count"],
                expected_chain_head=summary["haccp_evidence"]["chain_head"],
            )
            self.assertTrue(chain["valid"])
            self.assertEqual(schema["digital_twin_version"], "3.0.0")
            self.assertIn("flowchart LR", (first / "reference_plant.md").read_text())

    def test_invalid_config_fails_before_replacing_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._minimal_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["process"]["dt_s"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")
            target = root / "published"
            target.mkdir()
            sentinel = target / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(DigitalTwinConfigError, "must be numeric"):
                run_integrated(config_path, target)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
