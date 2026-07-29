#!/usr/bin/env python3
"""Focused tests for the v0.3 split diagnostic/operational report."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from ml_pipeline_common import sha256_file, write_checksums, write_json  # noqa: E402
from report_flowtwin_v03_development import (  # noqa: E402
    _atomic_output,
    _conditional_operational_row,
    _diagnostic_profile_signature,
    _diagnostic_row,
    _validate_tcn_audit,
)


def _scope(value: float, *, operational: bool = True) -> dict:
    result = {
        "rows": 100,
        "class_diagnosis": {"macro_f1": value},
        "event_detection": {"event_f1": value, "event_recall": value},
        "false_alarms": {"false_alarm_onsets_per_negative_hour": value},
        "unsafe_forward_l_before_first_post_effect_alarm": value,
        "conformal": {
            "empirical_coverage": value,
            "mean_prediction_set_size": 2.0,
            "singleton_fraction": 0.0,
        },
    }
    if operational:
        result.update(
            {
                "operational_event_detection": {
                    "event_f1": value,
                    "event_recall": value,
                },
                "operational_false_alarms": {
                    "false_alarm_onsets_per_negative_hour": value,
                },
                "operational_unsafe_forward_l_before_first_post_effect_alarm": value,
            }
        )
    return result


def _record(variant: str, *, operational: bool = True) -> dict:
    overall = _scope(0.7, operational=operational)
    return {
        "run_id": f"{variant}__seed-20260727",
        "variant": {"variant_id": variant},
        "seed": 20260727,
        "metrics": {
            "overall": overall,
            "by_split": {
                "test_id": _scope(0.71, operational=operational),
                "test_ood_profile": _scope(0.69, operational=operational),
            },
            "by_profile": {
                "P-ID": _scope(0.71, operational=operational),
                "P-OOD": _scope(0.69, operational=operational),
            },
            "profile_splits": {
                "P-ID": "test_id",
                "P-OOD": "test_ood_profile",
            },
            "ood_detection": {"auroc": 0.6, "aupr": 0.7, "fpr95": 0.9},
            "efficiency": {
                "parameter_count": 10,
                "training_wall_time_s": 2.0,
                "test_inference_rows_per_s": 100.0,
            },
        },
    }


class FlowTwinV03DevelopmentReportTests(unittest.TestCase):
    def test_diagnostic_table_labels_fitted_row_and_noncomparable_timing(self) -> None:
        row = _diagnostic_row(_record("tcn", operational=False))
        self.assertEqual(row["validation_fitted_row_event_f1"], 0.7)
        self.assertFalse(row["timing_comparable_across_variants"])
        self.assertEqual(row["statistical_role"], "descriptive_pooled_not_inferential")

    def test_conditional_operational_table_is_explicit(self) -> None:
        row = _conditional_operational_row(_record("flowtwin_guard"))
        self.assertEqual(row["statistical_role"], "conditional_on_validation_feasibility")
        self.assertEqual(row["maximum_profile_false_alarm_onsets_per_negative_hour"], 0.71)

    def test_profile_coverage_drift_is_rejected(self) -> None:
        first = _record("flowtwin_hybrid_v03_dev")
        second = _record("tcn", operational=False)
        second["metrics"]["by_profile"].pop("P-OOD")
        second["metrics"]["profile_splits"].pop("P-OOD")
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            _diagnostic_profile_signature([first, second])

    def test_tcn_no_operating_point_remains_null_and_test_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            run_dir = raw / "runs" / "tcn__seed-20260727"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            audit = root / "audit"
            audit.mkdir()
            full_audit = {
                "candidate_count": 75,
                "feasible_candidate_count": 0,
                "no_operating_point_reasons": ["recall"],
            }
            write_json(
                audit / "alarm_feasibility.json",
                {
                    "artifact_status": "complete",
                    "validation_gate_status": "no_operating_point",
                    "alarm_fit": {
                        "status": "no_operating_point",
                        "config": None,
                        "feasible_candidate_count": 0,
                        "full_audit": full_audit,
                    },
                },
            )
            write_json(
                audit / "validation_access_audit.json",
                {
                    "test_iterator_constructed": False,
                    "test_rows_seen": 0,
                    "opened_splits": ["validation_id"],
                },
            )
            write_json(
                audit / "checkpoint_lineage.json",
                {"raw_checkpoint": {"sha256": sha256_file(checkpoint)}},
            )
            write_checksums(
                audit,
                (
                    "alarm_feasibility.json",
                    "validation_access_audit.json",
                    "checkpoint_lineage.json",
                ),
            )
            tcn = {"root": raw}
            row, _source = _validate_tcn_audit(
                audit, tcn, {"run_id": "tcn__seed-20260727"}
            )
            self.assertEqual(row["validation_gate_status"], "no_operating_point")
            self.assertIsNone(row["operational_test_metrics"])
            self.assertFalse(row["operational_test_evaluated"])

    def test_nonempty_output_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                _atomic_output(output)


if __name__ == "__main__":
    unittest.main()
