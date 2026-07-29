#!/usr/bin/env python3
"""Tests for the validation-only FlowTwin v0.3 TCN alarm audit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import audit_flowtwin_v03_alarm as alarm_audit  # noqa: E402
from ml_pipeline_common import (  # noqa: E402
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
)


SERIALIZED_MODEL = {
    "kind": "test_tcn",
    "model": {"model_name": "tcn"},
    "training_options": {"train_nominal_observer": False},
    "evaluation_options": {
        "conformal_enabled": False,
        "abstention_enabled": True,
        "decision_policy": "validation_only_conformal",
    },
}
PROTOCOL = {
    "training": {"train_nominal_observer": False},
    "evaluation": {
        "conformal_enabled": False,
        "abstention_enabled": True,
        "decision_policy": "validation_only_conformal",
    },
}


def _validation_rows(*, detectable: bool = True) -> dict[str, object]:
    truth = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16)
    anomaly = (
        np.asarray([0.1, 0.1, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8])
        if detectable
        else np.full(8, 0.01)
    ).astype(np.float32)
    probability = np.column_stack([1.0 - anomaly, anomaly]).astype(np.float32)
    return {
        "probabilities": probability,
        "anomaly_probability": anomaly,
        "energies": np.zeros(8, dtype=np.float32),
        "class_target": truth.copy(),
        "anomaly_target": truth.copy(),
        "mode": np.zeros(8, dtype=np.float32),
        "time_s": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32),
        "step_dt_s": np.ones(8, dtype=np.float32),
        "effect_time_s": np.asarray(
            [np.nan, np.nan, 2.0, 2.0, np.nan, np.nan, 2.0, 2.0],
            dtype=np.float32,
        ),
        "detection_eligible": np.ones(8, dtype=np.bool_),
        "episode_index": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        "episode_vocabulary": ["E1", "E2"],
        "profile_index": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16),
        "profile_vocabulary": ["P1", "P2"],
        "counterfactual_group_index": np.asarray(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32
        ),
        "counterfactual_group_vocabulary": ["G1", "G2"],
        "split_index": np.zeros(8, dtype=np.int16),
        "split_vocabulary": ["validation_id"],
    }


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = root / "dataset"
        self.splits = root / "splits"
        self.cache = root / "cache"
        self.raw = root / "raw"
        self.output = root / "output"
        self.benchmark = root / "flowtwin_benchmark_contract.json"
        self.candidate = root / "flowtwin_v03_candidate_contract.json"
        self._write()

    def _write(self) -> None:
        self.dataset.mkdir()
        self.splits.mkdir()
        self.cache.mkdir()
        (self.raw / "runs" / "tcn__seed-20260727").mkdir(parents=True)
        self.benchmark.write_bytes(
            (MODULE_DIR / "flowtwin_benchmark_contract.json").read_bytes()
        )
        write_json(
            self.dataset / "dataset_manifest.json",
            {"dataset_version": "D2-ood-dev"},
        )
        write_checksums(self.dataset, ("dataset_manifest.json",))
        write_json(self.splits / "split_manifest.json", {"version": "test"})
        write_checksums(self.splits, ("split_manifest.json",))
        write_json(self.cache / "cache_manifest.json", {"version": "test"})
        write_checksums(self.cache, ("cache_manifest.json",))

        candidate = json.loads(
            (MODULE_DIR / "flowtwin_v03_candidate_contract.json").read_text(
                encoding="utf-8"
            )
        )
        candidate["base_benchmark_contract"]["sha256"] = sha256_file(self.benchmark)
        candidate["development_dataset"]["dataset_manifest_sha256"] = sha256_file(
            self.dataset / "dataset_manifest.json"
        )
        candidate["development_dataset"]["split_manifest_sha256"] = sha256_file(
            self.splits / "split_manifest.json"
        )
        candidate["development_dataset"]["cache_manifest_sha256"] = sha256_file(
            self.cache / "cache_manifest.json"
        )
        write_json(self.candidate, candidate)

        model = torch.nn.Linear(1, 1)
        checkpoint_path = (
            self.raw / "runs" / "tcn__seed-20260727" / "model.pt"
        )
        torch.save(
            {
                "benchmark_runner_version": alarm_audit.BENCHMARK_RUNNER_VERSION,
                "variant": alarm_audit.EXPECTED_VARIANT.to_dict(),
                "seed": 20260727,
                "serialized_model": SERIALIZED_MODEL,
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        run = {
            "run_id": "tcn__seed-20260727",
            "variant": alarm_audit.EXPECTED_VARIANT.to_dict(),
            "seed": 20260727,
            "model": SERIALIZED_MODEL,
            "training_protocol": PROTOCOL["training"],
            "evaluation_protocol": PROTOCOL["evaluation"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        write_json(
            self.raw / "runs" / "tcn__seed-20260727" / "run_record.json", run
        )
        write_json(self.raw / "standardizer.json", {"stub": "standardizer"})
        write_json(
            self.raw / "benchmark_config.json",
            {
                "benchmark_runner_version": alarm_audit.BENCHMARK_RUNNER_VERSION,
                "benchmark_contract_sha256": sha256_file(self.benchmark),
                "budget": candidate["training_budget"],
                "device": "cpu",
                "candidate_contract": None,
                "variants": [alarm_audit.EXPECTED_VARIANT.to_dict()],
            },
        )
        write_json(self.raw / "benchmark_results.json", {"runs": [run]})
        split_domain_audit = {
            "status": "pass",
            "episodes": 2,
            "profiles": 2,
            "counterfactual_groups": 2,
            "domain_episode_counts": {"ID": 2},
            "observed_ood_domains": [],
            "domain_contract_verified": True,
            "profile_table_domain_verified": True,
            "profile_parameter_ranges_verified": True,
        }
        write_json(self.raw / "split_domain_audit.json", split_domain_audit)
        write_json(
            self.raw / "benchmark_results.json",
            {"runs": [run], "split_domain_audit": split_domain_audit},
        )
        current = {
            name: sha256_file(MODULE_DIR / name)
            for name in alarm_audit.RUNTIME_SOURCE_NAMES
        }
        write_json(
            self.raw / "run_manifest.json",
            {
                "benchmark_runner_version": alarm_audit.BENCHMARK_RUNNER_VERSION,
                "benchmark_contract_source": {
                    "sha256": sha256_file(self.benchmark)
                },
                "dataset_manifest_sha256": sha256_file(
                    self.dataset / "dataset_manifest.json"
                ),
                "split_manifest_sha256": sha256_file(
                    self.splits / "split_manifest.json"
                ),
                "cache_manifest_sha256": sha256_file(
                    self.cache / "cache_manifest.json"
                ),
                "dataset_checksums_sha256": sha256_file(
                    self.dataset / "checksums.sha256"
                ),
                "split_checksums_sha256": sha256_file(
                    self.splits / "checksums.sha256"
                ),
                "split_policy": {"calibration": ["validation_id"]},
                "environment": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "device": "cpu",
                    "torch_num_threads": 1,
                    "deterministic_algorithms": True,
                },
                "provenance": {
                    **current,
                    "benchmark_contract": sha256_file(self.benchmark),
                },
            },
        )
        write_checksums(
            self.raw,
            (
                "benchmark_config.json",
                "benchmark_results.json",
                "run_manifest.json",
                "split_domain_audit.json",
                "standardizer.json",
                "runs/tcn__seed-20260727/model.pt",
                "runs/tcn__seed-20260727/run_record.json",
            ),
        )

    def rewrite_raw_checksums(self) -> None:
        paths = [
            line.split("  ", 1)[1]
            for line in (self.raw / "checksums.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        write_checksums(self.raw, paths)


class FlowTwinV03AlarmAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *, detectable: bool = True, output: Path | None = None):
        opened: list[tuple[str, ...]] = []

        def collect(_model, _context, accepted_splits):
            accepted = tuple(accepted_splits)
            opened.append(accepted)
            if any(str(name).startswith("test_") for name in accepted):
                raise AssertionError("test split was accessed")
            return deepcopy(_validation_rows(detectable=detectable)), 0.01

        def model_configs(_context, variant):
            self.assertEqual(variant, alarm_audit.EXPECTED_VARIANT)
            return torch.nn.Linear(1, 1), deepcopy(SERIALIZED_MODEL), deepcopy(PROTOCOL)

        with (
            patch.object(alarm_audit, "build_process_graph", return_value=object()),
            patch.object(alarm_audit, "load_target_contract", return_value=object()),
            patch.object(
                alarm_audit,
                "load_cache_manifest",
                return_value={"standardizer": {"stub": "standardizer"}},
            ),
            patch.object(
                alarm_audit.FeatureStandardizer,
                "from_dict",
                return_value=object(),
            ),
            patch.object(alarm_audit, "_model_configs", side_effect=model_configs),
            patch.object(alarm_audit, "_collect", side_effect=collect),
        ):
            result = alarm_audit.audit(
                raw_result=self.fixture.raw,
                dataset=self.fixture.dataset,
                splits=self.fixture.splits,
                cache=self.fixture.cache,
                output=output or self.fixture.output,
                benchmark_contract=self.fixture.benchmark,
                candidate_contract=self.fixture.candidate,
            )
        return result, opened

    def test_validation_only_selected_audit_is_atomic_and_checksummed(self) -> None:
        output, opened = self._run()
        self.assertEqual(opened, [("validation_id",)])
        self.assertEqual(verify_checksums(output), [])
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "alarm_feasibility.json",
                "checkpoint_lineage.json",
                "validation_access_audit.json",
                "calibration.json",
                "run_manifest.json",
                "checksums.sha256",
            },
        )
        result = json.loads((output / "alarm_feasibility.json").read_text())
        access = json.loads((output / "validation_access_audit.json").read_text())
        calibration = json.loads((output / "calibration.json").read_text())
        self.assertEqual(result["validation_gate_status"], "selected")
        self.assertGreater(result["alarm_fit"]["feasible_candidate_count"], 0)
        self.assertEqual(result["alarm_fit"]["candidate_count"], 75)
        self.assertEqual(access["opened_splits"], ["validation_id"])
        self.assertFalse(access["test_iterator_constructed"])
        self.assertEqual(access["test_rows_seen"], 0)
        self.assertEqual(access["test_rows_decoded_for_model_or_metrics"], 0)
        self.assertTrue(access["integrity_hashing_reads_declared_input_bytes"])
        self.assertFalse(access["prior_test_results_used_for_alarm_fit_or_selection"])
        self.assertEqual(
            calibration["operational_alarm"]["audit"]["fit_split"],
            "validation_id",
        )
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            self._run()

    def test_no_operating_point_is_persisted_with_full_rejections(self) -> None:
        output, opened = self._run(detectable=False)
        self.assertEqual(opened, [("validation_id",)])
        result = json.loads((output / "alarm_feasibility.json").read_text())
        calibration = json.loads((output / "calibration.json").read_text())
        self.assertEqual(result["validation_gate_status"], "no_operating_point")
        self.assertEqual(result["alarm_fit"]["feasible_candidate_count"], 0)
        self.assertIn(
            "minimum_event_recall_not_met",
            result["alarm_fit"]["no_operating_point_reasons"],
        )
        self.assertEqual(
            len(calibration["operational_alarm"]["audit"]["candidates"]),
            75,
        )
        self.assertEqual(verify_checksums(output), [])

    def test_tampered_checkpoint_fails_before_collection_or_publication(self) -> None:
        checkpoint = (
            self.fixture.raw
            / "runs"
            / "tcn__seed-20260727"
            / "model.pt"
        )
        with checkpoint.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._run()
        self.assertFalse(self.fixture.output.exists())

    def test_raw_source_provenance_mismatch_fails_closed(self) -> None:
        path = self.fixture.raw / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["provenance"]["flowtwin_guard/alarm.py"] = "0" * 64
        write_json(path, manifest)
        self.fixture.rewrite_raw_checksums()
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self._run()
        self.assertFalse(self.fixture.output.exists())

    def test_budget_mismatch_fails_before_collection(self) -> None:
        path = self.fixture.raw / "benchmark_config.json"
        config = json.loads(path.read_text())
        config["budget"]["window_size"] = 64
        write_json(path, config)
        self.fixture.rewrite_raw_checksums()
        with self.assertRaisesRegex(ValueError, "exact v0.3 candidate budget"):
            self._run()
        self.assertFalse(self.fixture.output.exists())

    def test_candidate_input_binding_mismatch_fails_closed(self) -> None:
        candidate = json.loads(self.fixture.candidate.read_text())
        candidate["development_dataset"]["cache_manifest_sha256"] = "0" * 64
        write_json(self.fixture.candidate, candidate)
        with self.assertRaisesRegex(ValueError, "artifact binding mismatch"):
            self._run()
        self.assertFalse(self.fixture.output.exists())

    def test_checkpoint_state_is_strictly_reloaded(self) -> None:
        checkpoint = (
            self.fixture.raw
            / "runs"
            / "tcn__seed-20260727"
            / "model.pt"
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved["state_dict"].pop("bias")
        torch.save(saved, checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        record_path = (
            self.fixture.raw
            / "runs"
            / "tcn__seed-20260727"
            / "run_record.json"
        )
        record = json.loads(record_path.read_text())
        record["checkpoint_sha256"] = checkpoint_sha
        write_json(record_path, record)
        results_path = self.fixture.raw / "benchmark_results.json"
        results = json.loads(results_path.read_text())
        results["runs"] = [record]
        write_json(results_path, results)
        self.fixture.rewrite_raw_checksums()
        with self.assertRaisesRegex(ValueError, "strict checkpoint state reload failed"):
            self._run()
        self.assertFalse(self.fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
