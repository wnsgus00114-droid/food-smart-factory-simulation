#!/usr/bin/env python3
"""Focused protocol tests for the FlowTwin benchmark orchestrator."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from run_flowtwin_benchmark import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_V03_CANDIDATE_CONTRACT,
    FULL_MODEL_NAME,
    V03_CANDIDATE_NAME,
    _fit_validation_calibration,
    _fit_taxonomy_hierarchical_calibration,
    _hierarchical_profile_bootstrap,
    _counterfactual_group_mode_calibration,
    _taxonomy_hierarchical_scale,
    _tier_assessment,
    _validate_split_domain_contract,
    _validate_v03_candidate_bindings,
    load_benchmark_contract,
    load_v03_candidate_contract,
    registered_variants,
    resolve_variants,
)
from ml_pipeline_common import sha256_file, sha256_json  # noqa: E402


class FlowTwinBenchmarkProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_benchmark_contract(DEFAULT_CONTRACT)

    def _budget(self) -> dict[str, object]:
        budget = copy.deepcopy(self.contract["training_budget"])
        budget["feature_set"] = self.contract["dataset_contract"]["feature_set"]
        budget["alpha"] = self.contract["calibration"]["conformal_alpha"]
        budget["ood_alpha"] = self.contract["calibration"]["ood_alpha"]
        return budget

    def test_contract_registers_full_baselines_and_nine_ablations(self) -> None:
        variants = registered_variants(self.contract)
        self.assertEqual(len(variants), 16)
        self.assertEqual(variants[0].variant_id, FULL_MODEL_NAME)
        self.assertNotIn(V03_CANDIDATE_NAME, {item.variant_id for item in variants})
        self.assertEqual(len([item for item in variants if item.family == "neural_baseline"]), 6)
        self.assertEqual(len([item for item in variants if item.family == "flowtwin_ablation"]), 9)
        self.assertEqual(resolve_variants(["all"], self.contract), variants)
        self.assertNotIn(
            V03_CANDIDATE_NAME,
            {item.variant_id for item in resolve_variants(["all"], self.contract)},
        )
        self.assertEqual(resolve_variants(["models"], self.contract), variants[:7])
        ablations = resolve_variants(["ablations"], self.contract)
        self.assertEqual(len(ablations), 10)
        self.assertEqual(ablations[0].variant_id, FULL_MODEL_NAME)
        with self.assertRaisesRegex(ValueError, "unknown --variants"):
            resolve_variants(["future_oracle_model"], self.contract)

    def test_explicit_v03_candidate_resolves_only_as_development(self) -> None:
        candidate = resolve_variants([V03_CANDIDATE_NAME], self.contract)
        self.assertEqual(len(candidate), 1)
        self.assertEqual(candidate[0].variant_id, V03_CANDIDATE_NAME)
        self.assertEqual(candidate[0].family, "experimental_candidate")
        self.assertEqual(candidate[0].model_name, V03_CANDIDATE_NAME)

        candidate_contract = load_v03_candidate_contract(
            DEFAULT_V03_CANDIDATE_CONTRACT
        )
        budget = copy.deepcopy(candidate_contract["training_budget"])
        assessment = _tier_assessment(
            candidate,
            budget,
            self.contract,
            ("test_id", "test_ood_profile"),
            torch.device("cpu"),
            {
                "domain_contract_verified": True,
                "profile_table_domain_verified": True,
                "profile_parameter_ranges_verified": True,
                "observed_ood_domains": ["OOD-profile"],
            },
        )
        self.assertEqual(assessment["tier"], "development")
        self.assertFalse(assessment["all_registered_variants"])
        self.assertEqual(assessment["extra_variants"], [V03_CANDIDATE_NAME])

    def test_v03_contract_discloses_opened_d2_and_window_96_gate(self) -> None:
        candidate = load_v03_candidate_contract(DEFAULT_V03_CANDIDATE_CONTRACT)
        self.assertEqual(candidate["candidate_id"], V03_CANDIDATE_NAME)
        self.assertEqual(candidate["status"], "development_only")
        self.assertTrue(candidate["d2_test_seen_before_freeze"])
        self.assertEqual(
            candidate["development_dataset"]["test_status"],
            "opened development set; never eligible for confirmatory evidence",
        )
        self.assertIn("newly generated", candidate["future_confirmatory_gate"])
        self.assertEqual(candidate["training_budget"]["window_size"], 96)
        gate = candidate["critical_delay_gate"]
        self.assertEqual(gate["fit_split"], "train_id")
        self.assertEqual(gate["minimum_exhaustive_per_edge_coverage"], 0.95)
        self.assertEqual(gate["minimum_sampled_train_per_edge_coverage"], 0.9)
        self.assertIn("post_hoc_design_disclosure", gate)

    def test_taxonomy_hierarchical_calibration_preserves_fault_mass(self) -> None:
        # N00/N01/N02 are non-fault states; classes 3..5 are faults.  The raw
        # candidate contract requires its binary fault probability to equal
        # the total mass of the fault-class partition before calibration.
        probabilities = np.asarray(
            [
                [0.60, 0.20, 0.10, 0.04, 0.03, 0.03],
                [0.10, 0.55, 0.15, 0.08, 0.07, 0.05],
                [0.05, 0.10, 0.15, 0.35, 0.20, 0.15],
                [0.04, 0.06, 0.10, 0.15, 0.40, 0.25],
                [0.15, 0.10, 0.05, 0.10, 0.15, 0.45],
                [0.20, 0.40, 0.10, 0.12, 0.10, 0.08],
            ],
            dtype=np.float64,
        )
        fault_indices = (3, 4, 5)
        anomaly = probabilities[:, fault_indices].sum(axis=1)
        validation = {
            "probabilities": probabilities,
            "anomaly_probability": anomaly,
            "class_target": np.asarray([0, 1, 3, 4, 5, 2], dtype=np.int64),
            "anomaly_target": np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64),
        }

        state, calibrated, calibrated_anomaly = (
            _fit_taxonomy_hierarchical_calibration(validation, fault_indices)
        )
        self.assertEqual(state["non_fault_class_indices"], [0, 1, 2])
        self.assertEqual(state["fault_class_indices"], [3, 4, 5])
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-7)
        np.testing.assert_allclose(
            calibrated[:, fault_indices].sum(axis=1),
            calibrated_anomaly,
            atol=1e-7,
        )

        directly_scaled, directly_scaled_anomaly = _taxonomy_hierarchical_scale(
            probabilities,
            anomaly,
            fault_indices,
            anomaly_temperature=1.3,
            non_fault_temperature=0.8,
            fault_temperature=1.7,
        )
        np.testing.assert_allclose(directly_scaled.sum(axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(
            directly_scaled[:, fault_indices].sum(axis=1),
            directly_scaled_anomaly,
            atol=1e-8,
        )

        incoherent = dict(validation)
        incoherent["anomaly_probability"] = anomaly.copy()
        incoherent["anomaly_probability"][0] += 0.02
        with self.assertRaisesRegex(ValueError, "hierarchy is incoherent"):
            _fit_taxonomy_hierarchical_calibration(incoherent, fault_indices)

    def test_v03_artifact_binding_rejects_any_budget_override(self) -> None:
        candidate = copy.deepcopy(
            load_v03_candidate_contract(DEFAULT_V03_CANDIDATE_CONTRACT)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            splits = root / "splits"
            cache = root / "cache"
            for directory in (dataset, splits, cache):
                directory.mkdir()
            benchmark_contract = root / "benchmark.json"
            candidate_contract = root / "candidate.json"
            benchmark_contract.write_text("{}\n", encoding="utf-8")
            candidate_contract.write_text("{}\n", encoding="utf-8")
            (dataset / "dataset_manifest.json").write_text(
                json.dumps({"dataset_version": "D2-ood-dev"}) + "\n",
                encoding="utf-8",
            )
            (splits / "split_manifest.json").write_text("{}\n", encoding="utf-8")
            (cache / "cache_manifest.json").write_text("{}\n", encoding="utf-8")

            candidate["base_benchmark_contract"]["sha256"] = sha256_file(
                benchmark_contract
            )
            development = candidate["development_dataset"]
            development["dataset_manifest_sha256"] = sha256_file(
                dataset / "dataset_manifest.json"
            )
            development["split_manifest_sha256"] = sha256_file(
                splits / "split_manifest.json"
            )
            development["cache_manifest_sha256"] = sha256_file(
                cache / "cache_manifest.json"
            )
            budget = copy.deepcopy(candidate["training_budget"])

            audit = _validate_v03_candidate_bindings(
                candidate,
                candidate_contract_path=candidate_contract,
                benchmark_contract_path=benchmark_contract,
                dataset=dataset,
                splits=splits,
                cache=cache,
                budget=budget,
            )
            self.assertEqual(audit["status"], "pass")
            self.assertTrue(audit["budget_exact"])
            self.assertFalse(audit["eligible_for_confirmatory_claims"])

            overridden = copy.deepcopy(budget)
            overridden["window_size"] = 64
            with self.assertRaisesRegex(
                ValueError, "fixed budget cannot be overridden.*window_size"
            ):
                _validate_v03_candidate_bindings(
                    candidate,
                    candidate_contract_path=candidate_contract,
                    benchmark_contract_path=benchmark_contract,
                    dataset=dataset,
                    splits=splits,
                    cache=cache,
                    budget=overridden,
                )

    def test_calibration_firewall_rejects_test_rows(self) -> None:
        collected = {
            "split_vocabulary": ["test_id"],
            "probabilities": np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
            "class_target": np.asarray([0, 1], dtype=np.int16),
            "anomaly_probability": np.asarray([0.1, 0.9], dtype=np.float32),
            "anomaly_target": np.asarray([0, 1], dtype=np.int16),
            "mode": np.zeros(2, dtype=np.float32),
            "energies": np.zeros(2, dtype=np.float32),
        }
        context = SimpleNamespace(budget={"alpha": 0.1, "ood_alpha": 0.01})
        with self.assertRaisesRegex(ValueError, "firewall violation"):
            _fit_validation_calibration(
                collected,
                context,
                {"conformal_enabled": False},
            )

    def test_profile_hierarchical_bootstrap_is_deterministic(self) -> None:
        variant = {
            "P01": {1: 0.9, 2: 0.8},
            "P02": {1: 0.7, 2: 0.6},
            "P03": {1: 0.5, 2: 0.4},
        }
        reference = {
            "P01": {1: 0.8, 2: 0.7},
            "P02": {1: 0.65, 2: 0.55},
            "P03": {1: 0.45, 2: 0.35},
        }
        first = _hierarchical_profile_bootstrap(
            variant, reference, repetitions=250, seed=12345
        )
        second = _hierarchical_profile_bootstrap(
            variant, reference, repetitions=250, seed=12345
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "reported")
        self.assertEqual(first["profiles"], 3)
        self.assertAlmostEqual(first["paired_mean_difference"], 2.0 / 30.0)
        self.assertIn("plant_profile_id", first["resampling"])

    def test_tier_never_calls_id_only_run_confirmatory(self) -> None:
        variants = registered_variants(self.contract)
        budget = self._budget()
        pilot = _tier_assessment(
            variants, budget, self.contract, ("test_id",), torch.device("cpu")
        )
        self.assertEqual(pilot["tier"], "pilot")
        self.assertFalse(pilot["has_test_ood_profile"])
        confirmatory = _tier_assessment(
            variants,
            budget,
            self.contract,
            ("test_id", "test_ood_profile"),
            torch.device("cpu"),
            {
                "domain_contract_verified": True,
                "profile_table_domain_verified": True,
                "profile_parameter_ranges_verified": True,
                "observed_ood_domains": ["OOD-profile"],
            },
        )
        self.assertEqual(confirmatory["tier"], "protocol_complete_synthetic")
        self.assertFalse(confirmatory["external_confirmatory_claim"])
        development = _tier_assessment(
            variants[:7], budget, self.contract, ("test_id",), torch.device("cpu")
        )
        self.assertEqual(development["tier"], "development")

        changed_alpha = copy.deepcopy(budget)
        changed_alpha["alpha"] = 0.2
        calibration_override = _tier_assessment(
            variants,
            changed_alpha,
            self.contract,
            ("test_id", "test_ood_profile"),
            torch.device("cpu"),
            {
                "domain_contract_verified": True,
                "profile_table_domain_verified": True,
                "profile_parameter_ranges_verified": True,
                "observed_ood_domains": ["OOD-profile"],
            },
        )
        self.assertEqual(calibration_override["tier"], "development")
        self.assertIn("alpha", calibration_override["budget_differences"])

    def test_group_mode_calibration_uses_block_maxima(self) -> None:
        validation = {
            "probabilities": np.asarray(
                [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]],
                dtype=np.float64,
            ),
            "class_target": np.asarray([0, 0, 0, 1]),
            "mode": np.asarray([0.0, 0.0, 1.0, 1.0]),
            "energies": np.asarray([1.0, 4.0, 2.0, 3.0]),
            "counterfactual_group_index": np.asarray([0, 0, 0, 0]),
            "profile_index": np.asarray([0, 0, 0, 0]),
            "counterfactual_group_vocabulary": ["G00"],
            "profile_vocabulary": ["P00"],
        }
        blocks, audit = _counterfactual_group_mode_calibration(validation)
        self.assertEqual(audit["calibration_blocks"], 2)
        self.assertEqual(audit["counterfactual_groups"], 1)
        self.assertEqual(blocks["energies"].tolist(), [4.0, 3.0])
        self.assertEqual(blocks["targets"].tolist(), [0, 1])

    def test_domain_contract_cross_checks_episode_truth(self) -> None:
        id_values = {"nominal_flow_l_h": 1.5, "raw_milk_temp_c": 5.5}
        ood_values = {"nominal_flow_l_h": 3.5, "raw_milk_temp_c": 5.5}
        fields = [
            "episode_id",
            "plant_profile_id",
            "counterfactual_group_id",
            "profile_config_hash",
            "noise_seed",
            "sampling_seed",
            "canonical_code",
            "domain",
            "split",
        ]
        rows = [
            {
                "episode_id": "E-ID",
                "plant_profile_id": "P-ID",
                "counterfactual_group_id": "G-ID",
                "profile_config_hash": sha256_json(id_values),
                "noise_seed": "1",
                "sampling_seed": "2",
                "canonical_code": "N00",
                "domain": "ID",
                "split": "test_id",
            },
            {
                "episode_id": "E-OOD",
                "plant_profile_id": "P-OOD",
                "counterfactual_group_id": "G-OOD",
                "profile_config_hash": sha256_json(ood_values),
                "noise_seed": "3",
                "sampling_seed": "4",
                "canonical_code": "N00",
                "domain": "OOD-profile",
                "split": "test_ood_profile",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            splits = root / "splits"
            dataset.mkdir()
            splits.mkdir()
            profile_fields = [
                "plant_profile_id",
                "profile_index",
                "profile_seed",
                "profile_config_hash",
                "domain",
                "nominal_flow_l_h",
                "raw_milk_temp_c",
            ]
            profile_rows = [
                {
                    "plant_profile_id": "P-ID",
                    "profile_index": "0",
                    "profile_seed": "11",
                    "profile_config_hash": sha256_json(id_values),
                    "domain": "ID",
                    **id_values,
                },
                {
                    "plant_profile_id": "P-OOD",
                    "profile_index": "1",
                    "profile_seed": "12",
                    "profile_config_hash": sha256_json(ood_values),
                    "domain": "OOD-profile",
                    **ood_values,
                },
            ]
            with (dataset / "profiles.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=profile_fields)
                writer.writeheader()
                writer.writerows(profile_rows)
            with (dataset / "episodes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields[:-1])
                writer.writeheader()
                writer.writerows(
                    [{key: value for key, value in row.items() if key != "split"} for row in rows]
                )
            with (splits / "episode_splits.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            (dataset / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "domain_contract": {
                            "version": "1.0.0",
                            "id_domain": "ID",
                            "ood_domains": ["OOD-profile"],
                            "generation_policy": {"policy": "held-out profile ranges"},
                            "profile_parameter_ranges": {
                                "ID": {
                                    "nominal_flow_l_h": [1, 2],
                                    "raw_milk_temp_c": [5, 6],
                                },
                                "OOD-profile": {
                                    "nominal_flow_l_h": [3, 4],
                                    "raw_milk_temp_c": [5, 6],
                                },
                            },
                            "source_contract": {
                                "filename": "ood_profile_contract.json",
                                "sha256": "a" * 64,
                            },
                            "synthetic_only": True,
                        },
                        "provenance": {"ood_profile_contract.json": "a" * 64},
                    }
                ),
                encoding="utf-8",
            )
            audit = _validate_split_domain_contract(dataset, splits)
            self.assertTrue(audit["domain_contract_verified"])
            self.assertTrue(audit["profile_table_domain_verified"])
            self.assertTrue(audit["profile_parameter_ranges_verified"])
            self.assertEqual(audit["observed_ood_domains"], ["OOD-profile"])

            rows[1]["domain"] = "ID"
            with (splits / "episode_splits.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                _validate_split_domain_contract(dataset, splits)

            rows[1]["domain"] = "OOD-profile"
            profile_rows[1]["nominal_flow_l_h"] = 1.5
            fake_values = {"nominal_flow_l_h": 1.5, "raw_milk_temp_c": 5.5}
            fake_hash = sha256_json(fake_values)
            profile_rows[1]["profile_config_hash"] = fake_hash
            rows[1]["profile_config_hash"] = fake_hash
            with (dataset / "profiles.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=profile_fields)
                writer.writeheader()
                writer.writerows(profile_rows)
            with (dataset / "episodes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields[:-1])
                writer.writeheader()
                writer.writerows(
                    [
                        {key: value for key, value in row.items() if key != "split"}
                        for row in rows
                    ]
                )
            with (splits / "episode_splits.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "outside declared"):
                _validate_split_domain_contract(dataset, splits)


if __name__ == "__main__":
    unittest.main()
