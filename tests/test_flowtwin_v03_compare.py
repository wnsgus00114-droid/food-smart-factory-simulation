#!/usr/bin/env python3
"""Tests for the fail-closed FlowTwin v0.3 shard comparison."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from compare_flowtwin_v03 import compare  # noqa: E402
from ml_pipeline_common import (  # noqa: E402
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
)


PLANNED = (
    "flowtwin_hybrid_v03_dev",
    "flowtwin_guard",
    "tcn",
    "dspr_diagnostic_adaptation",
)
SOURCE_FILES = (
    "benchmark_config.json",
    "benchmark_contract.json",
    "benchmark_results.json",
    "candidate_binding_audit.json",
    "candidate_contract.json",
    "run_manifest.json",
)


def _refresh_checksums(root: Path) -> None:
    artifacts = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    write_checksums(root, artifacts)


def _scope_metrics(value: float) -> dict[str, Any]:
    return {
        "rows": 100,
        "class_diagnosis": {"macro_f1": value},
        "event_detection": {
            "event_f1": value - 0.01,
            "event_recall": value - 0.02,
            "horizon_penalized_latency_mean_s": 10.0 + value,
        },
        "false_alarms": {
            "false_alarm_onsets_per_negative_hour": 3.0 + value,
        },
        "unsafe_forward_l_before_first_post_effect_alarm": 100.0 + value,
        "operational_event_detection": {
            "event_f1": value - 0.03,
            "event_recall": value - 0.04,
            "horizon_penalized_latency_mean_s": 12.0 + value,
        },
        "operational_false_alarms": {
            "false_alarm_onsets_per_negative_hour": 2.0 + value,
        },
        "operational_unsafe_forward_l_before_first_post_effect_alarm": 90.0
        + value,
        "conformal": {
            "empirical_coverage": 0.9 + value / 100.0,
            "mean_prediction_set_size": 2.0 + value,
        },
        "selective_risk": {"area_under_risk_coverage": 0.2 + value / 100.0},
    }


def _run_record(variant_id: str, seed: int, value: float) -> dict[str, Any]:
    overall = _scope_metrics(value)
    id_metrics = _scope_metrics(value + 0.01)
    ood_metrics = _scope_metrics(value - 0.01)
    return {
        "run_id": f"{variant_id}-seed-{seed}",
        "variant": {
            "variant_id": variant_id,
            "family": "synthetic_test",
            "model_name": variant_id,
            "ablation": None,
        },
        "seed": seed,
        "metrics": {
            "overall": overall,
            "by_split": {
                "test_id": id_metrics,
                "test_ood_profile": ood_metrics,
            },
            "by_profile": {
                "P-ID": id_metrics,
                "P-OOD": ood_metrics,
            },
            "profile_splits": {
                "P-ID": "test_id",
                "P-OOD": "test_ood_profile",
            },
            "ood_detection": {
                "auroc": value,
                "aupr": value - 0.05,
                "fpr95": 1.0 - value,
            },
            "efficiency": {
                "parameter_count": 1000.0 + value,
                "trainable_parameter_count_before_observer_freeze": 1100.0
                + value,
                "training_wall_time_s": 20.0 + value,
                "validation_inference_rows_per_s": 500.0 + value,
                "test_inference_rows_per_s": 600.0 + value,
            },
        },
    }


def _candidate_contract() -> dict[str, Any]:
    return {
        "candidate_contract_version": "0.1.0",
        "candidate_id": PLANNED[0],
        "status": "development_only",
        "frozen_before_this_execution": True,
        "d2_test_seen_before_freeze": True,
        "base_benchmark_contract": {
            "version": "test",
            "sha256": "b" * 64,
        },
        "development_dataset": {
            "dataset_manifest_sha256": "1" * 64,
            "split_manifest_sha256": "2" * 64,
            "cache_manifest_sha256": "5" * 64,
            "test_status": (
                "opened development set; never eligible for confirmatory evidence"
            )
        },
        "training_budget": {
            "seeds": [20260727],
            "window_size": 96,
            "diagnostic_epochs": 10,
        },
        "planned_comparison_variants": list(PLANNED),
        "claim_boundary": "Synthetic opened-D2 development evidence only.",
        "future_confirmatory_gate": "Use a newly sealed independent test set.",
    }


def _tier() -> dict[str, Any]:
    return {"tier": "development", "external_confirmatory_claim": False}


def _write_source(
    root: Path,
    variants: Sequence[str],
    *,
    binding_overrides: Mapping[str, str] | None = None,
    budget: Mapping[str, Any] | None = None,
) -> None:
    root.mkdir()
    candidate_sha = "a" * 64
    benchmark_sha = "b" * 64
    artifact_hashes = {
        "dataset_manifest_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64,
        "dataset_checksums_sha256": "3" * 64,
        "split_checksums_sha256": "4" * 64,
        "cache_manifest_sha256": "5" * 64,
    }
    artifact_hashes.update(binding_overrides or {})
    contract = _candidate_contract()
    actual_budget = dict(budget or contract["training_budget"])
    runs = [
        _run_record(variant, 20260727, 0.7 + index / 100.0)
        for index, variant in enumerate(variants)
    ]
    binding_checks = {
        "base_benchmark_contract_sha256": {
            "expected": benchmark_sha,
            "actual": benchmark_sha,
        },
        "dataset_manifest_sha256": {
            "expected": artifact_hashes["dataset_manifest_sha256"],
            "actual": artifact_hashes["dataset_manifest_sha256"],
        },
        "split_manifest_sha256": {
            "expected": artifact_hashes["split_manifest_sha256"],
            "actual": artifact_hashes["split_manifest_sha256"],
        },
        "cache_manifest_sha256": {
            "expected": artifact_hashes["cache_manifest_sha256"],
            "actual": artifact_hashes["cache_manifest_sha256"],
        },
    }
    candidate_binding = {
        "status": "pass",
        "budget_exact": True,
        "candidate_contract_sha256": candidate_sha,
        "d2_test_seen_before_freeze": True,
        "evidence_status": "post_hoc_development_only",
        "eligible_for_confirmatory_claims": False,
        "checks": binding_checks,
    }
    config = {
        "benchmark_runner_version": "0.3.0",
        "benchmark_contract_sha256": benchmark_sha,
        "candidate_contract": {
            "sha256": candidate_sha,
            "status": "post_hoc_development_only",
        },
        "budget": actual_budget,
        "variants": [{"variant_id": variant} for variant in variants],
        "device": "cpu",
        "cache": "/fixed/cache",
        "save_predictions": False,
        "test_splits": ["test_id", "test_ood_profile"],
        "train_validation_test_firewall": "fixed test firewall",
        "tier_assessment": _tier(),
    }
    benchmark_contract = {
        "benchmark_contract_version": "test",
        "aggregation": {
            "bootstrap_repetitions": 20,
            "bootstrap_seed": 19,
        },
    }
    results = {
        "tier_assessment": _tier(),
        "candidate_binding_audit": candidate_binding,
        "candidate_claim_boundary": contract["claim_boundary"],
        "runs": runs,
    }
    manifest = {
        **artifact_hashes,
        "benchmark_contract_source": {"sha256": benchmark_sha},
        "candidate_contract_source": {
            "sha256": candidate_sha,
            "d2_test_seen_before_freeze": True,
            "eligible_for_confirmatory_claims": False,
        },
        "benchmark_runner_version": "0.3.0",
        "benchmark_contract_version": "test",
        "model_version": "test",
        "hybrid_model_version": "test",
        "alarm_policy_version": "test",
        "baseline_version": "test",
        "ablation_version": "test",
        "metrics_version": "test",
        "environment": {
            "python": "test",
            "torch": "test",
            "numpy": "test",
            "device": "cpu",
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
        },
        "input_contract": {"feature_set": "test"},
        "split_policy": {
            "training": ["train_id"],
            "calibration": ["validation_id"],
            "evaluation": ["test_id", "test_ood_profile"],
        },
        "statistical_unit": {
            "inferential": "plant_profile_id",
            "technical_repeat": "training seed",
        },
        "provenance": {
            "run_flowtwin_benchmark.py": sha256_file(
                MODULE_DIR / "run_flowtwin_benchmark.py"
            ),
            "shared_source.py": "6" * 64,
        },
        "tier_assessment": _tier(),
        "candidate_binding_audit": candidate_binding,
        "candidate_claim_boundary": contract["claim_boundary"],
    }
    write_json(root / "benchmark_config.json", config)
    write_json(root / "benchmark_contract.json", benchmark_contract)
    write_json(root / "benchmark_results.json", results)
    write_json(root / "candidate_binding_audit.json", candidate_binding)
    write_json(root / "candidate_contract.json", contract)
    write_json(root / "run_manifest.json", manifest)
    for record in runs:
        run_dir = root / "runs" / record["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "model.pt").write_bytes(b"synthetic checkpoint")
        write_json(run_dir / "calibration.json", {"fit_split": "validation_id"})
        write_json(run_dir / "metrics.json", record["metrics"])
        write_json(run_dir / "training_history.json", {"diagnostic": []})
        record.update(
            {
                "model": {"kind": "synthetic_test"},
                "training_protocol": {"epochs": 1},
                "evaluation_protocol": {"fit_split": "validation_id"},
                "checkpoint_sha256": sha256_file(run_dir / "model.pt"),
                "calibration_sha256": sha256_file(run_dir / "calibration.json"),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                "training_history_sha256": sha256_file(
                    run_dir / "training_history.json"
                ),
            }
        )
        write_json(run_dir / "run_record.json", record)
    # Embedded records are finalized only after their run-artifact hashes exist.
    results["runs"] = runs
    write_json(root / "benchmark_results.json", results)
    _refresh_checksums(root)


class FlowTwinV03CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate"
        self.comparator = self.root / "comparator"
        _write_source(self.candidate, [PLANNED[0]])
        _write_source(self.comparator, PLANNED[1:])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_success_publishes_four_model_paired_aggregation_atomically(self) -> None:
        output = compare(self.candidate, self.comparator, self.root / "combined")
        self.assertEqual(verify_checksums(output), [])
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "comparison_results.json",
                "compact_table.csv",
                "run_manifest.json",
                "checksums.sha256",
            },
        )
        results = json.loads(
            (output / "comparison_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(results["planned_comparison_variants"], list(PLANNED))
        self.assertEqual(
            [run["variant"]["variant_id"] for run in results["runs"]],
            list(PLANNED),
        )
        self.assertEqual(len(results["compact_table"]), 4)
        self.assertTrue(
            all(
                row["statistical_role"] == "descriptive_pooled_not_inferential"
                for row in results["compact_table"]
            )
        )
        self.assertEqual(
            set(results["candidate_paired_profile_comparisons"]),
            set(PLANNED[1:]),
        )
        self.assertEqual(
            results["candidate_paired_profile_comparisons"][PLANNED[1]][
                "class_macro_f1"
            ]["difference"],
            "candidate_minus_comparator",
        )
        self.assertEqual(
            set(results["aggregation"]["paired_profile_comparisons"]),
            {PLANNED[0], PLANNED[2], PLANNED[3]},
        )
        with (output / "compact_table.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual([row["variant_id"] for row in table], list(PLANNED))
        self.assertIn("overall_class_macro_f1", table[0])
        self.assertIn("operational_event_f1", table[0])
        self.assertIn("ood_auroc", table[0])
        self.assertIn("test_inference_rows_per_s", table[0])
        self.assertEqual(
            table[0]["statistical_role"],
            "descriptive_pooled_not_inferential",
        )

        manifest = json.loads(
            (output / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["eligible_for_confirmatory_claims"])
        self.assertEqual(
            manifest["source_artifacts"]["candidate"][
                "benchmark_results_sha256"
            ],
            sha256_file(self.candidate / "benchmark_results.json"),
        )
        self.assertEqual(
            manifest["aggregation_runner_source"]["sha256"],
            sha256_file(MODULE_DIR / "run_flowtwin_benchmark.py"),
        )
        self.assertEqual(
            manifest["merger_source"]["sha256"],
            sha256_file(MODULE_DIR / "compare_flowtwin_v03.py"),
        )

    def test_checksum_failure_is_rejected_without_output(self) -> None:
        with (self.comparator / "benchmark_results.json").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("\n")
        output = self.root / "checksum-failure"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            compare(self.candidate, self.comparator, output)
        self.assertFalse(output.exists())

    def test_binding_mismatch_is_rejected(self) -> None:
        alternate = self.root / "different-binding"
        _write_source(
            alternate,
            PLANNED[1:],
            binding_overrides={"dataset_manifest_sha256": "f" * 64},
        )
        with self.assertRaisesRegex(
            ValueError, "candidate contract dataset_manifest_sha256 mismatch"
        ):
            compare(self.candidate, alternate, self.root / "binding-failure")

    def test_variant_union_mismatch_is_rejected(self) -> None:
        alternate = self.root / "wrong-variant"
        _write_source(alternate, [PLANNED[1], PLANNED[2], "unplanned_model"])
        with self.assertRaisesRegex(ValueError, "planned comparators"):
            compare(self.candidate, alternate, self.root / "variant-failure")

    def test_budget_contract_is_fail_closed(self) -> None:
        alternate = self.root / "wrong-budget"
        _write_source(
            alternate,
            PLANNED[1:],
            budget={"seeds": [20260727], "window_size": 64, "diagnostic_epochs": 10},
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            compare(self.candidate, alternate, self.root / "budget-failure")

    def test_seed_contract_is_fail_closed(self) -> None:
        alternate = self.root / "wrong-seed"
        _write_source(alternate, PLANNED[1:])
        results_path = alternate / "benchmark_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        for record in results["runs"]:
            record["seed"] = 123
            write_json(
                alternate / "runs" / record["run_id"] / "run_record.json",
                record,
            )
        write_json(results_path, results)
        _refresh_checksums(alternate)
        with self.assertRaisesRegex(ValueError, "seed set"):
            compare(self.candidate, alternate, self.root / "seed-failure")

    def test_execution_provenance_drift_is_rejected(self) -> None:
        manifest_path = self.comparator / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["environment"]["device"] = "mps"
        write_json(manifest_path, manifest)
        _refresh_checksums(self.comparator)
        with self.assertRaisesRegex(ValueError, "execution mismatch: environment"):
            compare(self.candidate, self.comparator, self.root / "env-failure")

    def test_contract_declared_binding_is_checked_directly(self) -> None:
        for source in (self.candidate, self.comparator):
            path = source / "candidate_contract.json"
            contract = json.loads(path.read_text(encoding="utf-8"))
            contract["development_dataset"]["dataset_manifest_sha256"] = "f" * 64
            write_json(path, contract)
            _refresh_checksums(source)
        with self.assertRaisesRegex(
            ValueError, "candidate contract dataset_manifest_sha256 mismatch"
        ):
            compare(self.candidate, self.comparator, self.root / "contract-failure")

    def test_missing_checksummed_run_artifact_is_rejected(self) -> None:
        record = json.loads(
            (self.comparator / "benchmark_results.json").read_text(encoding="utf-8")
        )["runs"][0]
        (self.comparator / "runs" / record["run_id"] / "model.pt").unlink()
        _refresh_checksums(self.comparator)
        with self.assertRaisesRegex(ValueError, "run artifact is not checksummed"):
            compare(self.candidate, self.comparator, self.root / "artifact-failure")

    def test_profile_coverage_drift_is_rejected(self) -> None:
        results_path = self.comparator / "benchmark_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        record = results["runs"][0]
        record["metrics"]["by_profile"].pop("P-OOD")
        record["metrics"]["profile_splits"].pop("P-OOD")
        run_dir = self.comparator / "runs" / record["run_id"]
        write_json(run_dir / "metrics.json", record["metrics"])
        record["metrics_sha256"] = sha256_file(run_dir / "metrics.json")
        write_json(run_dir / "run_record.json", record)
        write_json(results_path, results)
        _refresh_checksums(self.comparator)
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            compare(self.candidate, self.comparator, self.root / "profile-failure")

    def test_nonempty_output_is_never_overwritten(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            compare(self.candidate, self.comparator, output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
