#!/usr/bin/env python3
"""Fail-closed merger for the two matched FlowTwin v0.3 result shards.

The v0.3 development protocol is intentionally runnable as two shards: a
candidate-only run and a comparator-only run.  This tool publishes a combined
comparison only when the shards have identical provenance and jointly contain
exactly the variants frozen in the candidate contract.  It never upgrades the
opened D2 experiment to confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from ml_pipeline_common import sha256_file, sha256_json, write_checksums, write_json
from run_flowtwin_benchmark import (
    ENDPOINTS,
    _aggregate_results,
    _hierarchical_profile_bootstrap,
    _nested_number,
    _stable_bootstrap_seed,
)


COMPARISON_VERSION = "0.2.0"
MERGER_SOURCE = Path(__file__).resolve()
RUNNER_SOURCE = MERGER_SOURCE.parent / "run_flowtwin_benchmark.py"
EXECUTION_VERSION_FIELDS = (
    "benchmark_runner_version",
    "benchmark_contract_version",
    "model_version",
    "hybrid_model_version",
    "alarm_policy_version",
    "baseline_version",
    "ablation_version",
    "metrics_version",
)
EXECUTION_POLICY_FIELDS = (
    "environment",
    "input_contract",
    "split_policy",
    "statistical_unit",
    "provenance",
)
REQUIRED_SOURCE_ARTIFACTS = (
    "benchmark_config.json",
    "benchmark_contract.json",
    "benchmark_results.json",
    "candidate_binding_audit.json",
    "candidate_contract.json",
    "run_manifest.json",
)
OUTPUT_ARTIFACTS = (
    "comparison_results.json",
    "compact_table.csv",
    "run_manifest.json",
)


COMPACT_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("overall_class_macro_f1", ("metrics", "overall", "class_diagnosis", "macro_f1")),
    ("id_class_macro_f1", ("metrics", "by_split", "test_id", "class_diagnosis", "macro_f1")),
    ("ood_class_macro_f1", ("metrics", "by_split", "test_ood_profile", "class_diagnosis", "macro_f1")),
    ("raw_event_f1", ("metrics", "overall", "event_detection", "event_f1")),
    ("raw_event_recall", ("metrics", "overall", "event_detection", "event_recall")),
    (
        "raw_horizon_penalized_latency_mean_s",
        ("metrics", "overall", "event_detection", "horizon_penalized_latency_mean_s"),
    ),
    (
        "raw_false_alarm_onsets_per_negative_hour",
        ("metrics", "overall", "false_alarms", "false_alarm_onsets_per_negative_hour"),
    ),
    (
        "raw_unsafe_forward_l_before_detection",
        ("metrics", "overall", "unsafe_forward_l_before_first_post_effect_alarm"),
    ),
    (
        "operational_event_f1",
        ("metrics", "overall", "operational_event_detection", "event_f1"),
    ),
    (
        "operational_event_recall",
        ("metrics", "overall", "operational_event_detection", "event_recall"),
    ),
    (
        "operational_horizon_penalized_latency_mean_s",
        (
            "metrics",
            "overall",
            "operational_event_detection",
            "horizon_penalized_latency_mean_s",
        ),
    ),
    (
        "operational_false_alarm_onsets_per_negative_hour",
        (
            "metrics",
            "overall",
            "operational_false_alarms",
            "false_alarm_onsets_per_negative_hour",
        ),
    ),
    (
        "operational_unsafe_forward_l_before_detection",
        (
            "metrics",
            "overall",
            "operational_unsafe_forward_l_before_first_post_effect_alarm",
        ),
    ),
    ("ood_auroc", ("metrics", "ood_detection", "auroc")),
    ("ood_aupr", ("metrics", "ood_detection", "aupr")),
    ("ood_fpr95", ("metrics", "ood_detection", "fpr95")),
    (
        "conformal_coverage",
        ("metrics", "overall", "conformal", "empirical_coverage"),
    ),
    (
        "conformal_mean_prediction_set_size",
        ("metrics", "overall", "conformal", "mean_prediction_set_size"),
    ),
    ("parameter_count", ("metrics", "efficiency", "parameter_count")),
    (
        "trainable_parameter_count_before_observer_freeze",
        (
            "metrics",
            "efficiency",
            "trainable_parameter_count_before_observer_freeze",
        ),
    ),
    (
        "training_wall_time_s",
        ("metrics", "efficiency", "training_wall_time_s"),
    ),
    (
        "validation_inference_rows_per_s",
        ("metrics", "efficiency", "validation_inference_rows_per_s"),
    ),
    (
        "test_inference_rows_per_s",
        ("metrics", "efficiency", "test_inference_rows_per_s"),
    ),
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _strict_checksum_records(directory: Path) -> dict[str, str]:
    """Verify every declared source checksum and reject ambiguous records."""

    root = Path(directory).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"checksums.sha256 is missing: {root}")
    records: dict[str, str] = {}
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
        ):
            raise ValueError(
                f"malformed checksum record at {checksum_path}:{line_number}"
            )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"checksum path escapes source directory: {relative!r}")
        if relative in records:
            raise ValueError(f"duplicate checksum path: {relative}")
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"checksum target escapes source directory: {relative!r}"
            ) from exc
        if not target.is_file():
            raise ValueError(f"missing checksummed file: {relative}")
        actual = sha256_file(target)
        if actual != digest:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {digest}, got {actual}"
            )
        records[relative] = digest
    if not records:
        raise ValueError(f"checksums.sha256 is empty: {root}")
    missing = sorted(set(REQUIRED_SOURCE_ARTIFACTS) - set(records))
    if missing:
        raise ValueError(f"required source artifacts are not checksummed: {missing}")
    return records


def _load_source(directory: Path, role: str) -> dict[str, Any]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"{role} result directory does not exist: {root}")
    checksums = _strict_checksum_records(root)
    return {
        "role": role,
        "root": root,
        "checksums": checksums,
        "config": _json_object(root / "benchmark_config.json"),
        "contract": _json_object(root / "benchmark_contract.json"),
        "results": _json_object(root / "benchmark_results.json"),
        "candidate_binding": _json_object(root / "candidate_binding_audit.json"),
        "candidate_contract": _json_object(root / "candidate_contract.json"),
        "manifest": _json_object(root / "run_manifest.json"),
    }


def _safe_run_directory(source: Mapping[str, Any], run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"{source['role']} has unsafe run_id {run_id!r}")
    root = Path(source["root"])
    run_dir = (root / "runs" / run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{source['role']} run directory escapes source") from exc
    return run_dir


def _verify_run_artifacts(
    source: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    """Require every embedded run to be backed by checksummed artifacts."""

    save_predictions = source["config"].get("save_predictions")
    if not isinstance(save_predictions, bool):
        raise ValueError(f"{source['role']} save_predictions must be boolean")
    declared = set(source["checksums"])
    for index, record in enumerate(records):
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            raise ValueError(f"{source['role']} run {index} lacks run_id")
        run_dir = _safe_run_directory(source, run_id)
        required = [
            "model.pt",
            "calibration.json",
            "metrics.json",
            "training_history.json",
            "run_record.json",
        ]
        if save_predictions:
            required.append("test_predictions.npz")
        for name in required:
            path = run_dir / name
            relative = str(path.relative_to(source["root"]))
            if relative not in declared:
                raise ValueError(
                    f"{source['role']} run artifact is not checksummed: {relative}"
                )
            if not path.is_file():
                raise ValueError(f"{source['role']} run artifact is missing: {relative}")

        artifact_record = _json_object(run_dir / "run_record.json")
        artifact_metrics = _json_object(run_dir / "metrics.json")
        if artifact_record != record:
            raise ValueError(
                f"{source['role']} embedded and artifact run records differ: {run_id}"
            )
        if artifact_metrics != record.get("metrics"):
            raise ValueError(
                f"{source['role']} embedded and artifact metrics differ: {run_id}"
            )
        digest_fields = {
            "checkpoint_sha256": "model.pt",
            "calibration_sha256": "calibration.json",
            "metrics_sha256": "metrics.json",
            "training_history_sha256": "training_history.json",
        }
        for field, name in digest_fields.items():
            actual = sha256_file(run_dir / name)
            if record.get(field) != actual:
                raise ValueError(
                    f"{source['role']} {run_id} {field} does not bind {name}"
                )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _variant_id(record: Mapping[str, Any], label: str) -> str:
    variant = _require_mapping(record.get("variant"), f"{label}.variant")
    value = variant.get("variant_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.variant.variant_id must be a non-empty string")
    return value


def _source_variant_runs(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    raw_runs = source["results"].get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"{source['role']} benchmark_results.runs must be non-empty")
    runs: list[dict[str, Any]] = []
    seeds_by_variant: dict[str, set[int]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    for index, raw_record in enumerate(raw_runs):
        if not isinstance(raw_record, dict):
            raise ValueError(f"{source['role']} run {index} must be an object")
        variant_id = _variant_id(raw_record, f"{source['role']}.runs[{index}]")
        seed = raw_record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{source['role']} run {index} seed must be an integer")
        pair = (variant_id, seed)
        if pair in seen_pairs:
            raise ValueError(f"duplicate variant/seed run in {source['role']}: {pair}")
        seen_pairs.add(pair)
        seeds_by_variant.setdefault(variant_id, set()).add(seed)
        runs.append(raw_record)
    return runs, seeds_by_variant


def _config_variant_ids(source: Mapping[str, Any]) -> tuple[str, ...]:
    variants = source["config"].get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{source['role']} benchmark_config.variants must be non-empty")
    result: list[str] = []
    for index, variant in enumerate(variants):
        mapping = _require_mapping(
            variant, f"{source['role']}.benchmark_config.variants[{index}]"
        )
        variant_id = mapping.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError(f"invalid configured variant in {source['role']}")
        result.append(variant_id)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate configured variant in {source['role']}")
    return tuple(result)


def _candidate_source_sha(source: Mapping[str, Any]) -> str:
    candidate_source = _require_mapping(
        source["manifest"].get("candidate_contract_source"),
        f"{source['role']}.run_manifest.candidate_contract_source",
    )
    sha = candidate_source.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"{source['role']} candidate contract SHA is invalid")
    if candidate_source.get("d2_test_seen_before_freeze") is not True:
        raise ValueError(f"{source['role']} does not disclose opened D2")
    if candidate_source.get("eligible_for_confirmatory_claims") is not False:
        raise ValueError(f"{source['role']} incorrectly permits confirmatory claims")
    config_source = _require_mapping(
        source["config"].get("candidate_contract"),
        f"{source['role']}.benchmark_config.candidate_contract",
    )
    if config_source.get("sha256") != sha:
        raise ValueError(f"{source['role']} candidate contract SHA binding mismatch")
    binding = source["candidate_binding"]
    if binding.get("candidate_contract_sha256") != sha:
        raise ValueError(f"{source['role']} candidate binding SHA mismatch")
    return sha


def _validate_development_state(source: Mapping[str, Any]) -> None:
    role = str(source["role"])
    candidate = source["candidate_contract"]
    if candidate.get("status") != "development_only":
        raise ValueError(f"{role} candidate contract is not development_only")
    if candidate.get("d2_test_seen_before_freeze") is not True:
        raise ValueError(f"{role} candidate contract does not disclose opened D2")
    development = _require_mapping(
        candidate.get("development_dataset"),
        f"{role}.candidate_contract.development_dataset",
    )
    if development.get("test_status") != (
        "opened development set; never eligible for confirmatory evidence"
    ):
        raise ValueError(f"{role} D2 test status disclosure is invalid")
    binding = source["candidate_binding"]
    expected_binding = {
        "status": "pass",
        "budget_exact": True,
        "d2_test_seen_before_freeze": True,
        "evidence_status": "post_hoc_development_only",
        "eligible_for_confirmatory_claims": False,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise ValueError(f"{role} candidate binding field {key!r} is invalid")
    if source["results"].get("candidate_binding_audit") != binding:
        raise ValueError(f"{role} result/candidate-binding artifacts differ")
    if source["manifest"].get("candidate_binding_audit") != binding:
        raise ValueError(f"{role} manifest/candidate-binding artifacts differ")
    for document_name in ("config", "results", "manifest"):
        document = source[document_name]
        tier = _require_mapping(
            document.get("tier_assessment"),
            f"{role}.{document_name}.tier_assessment",
        )
        if tier.get("tier") != "development" or tier.get(
            "external_confirmatory_claim"
        ) is not False:
            raise ValueError(f"{role} {document_name} is not development-only")
    claim = candidate.get("claim_boundary")
    if not isinstance(claim, str) or not claim:
        raise ValueError(f"{role} candidate claim boundary is missing")
    if source["results"].get("candidate_claim_boundary") != claim:
        raise ValueError(f"{role} result claim boundary does not match its contract")
    if source["manifest"].get("candidate_claim_boundary") != claim:
        raise ValueError(f"{role} manifest claim boundary does not match its contract")


def _binding_fingerprint(source: Mapping[str, Any]) -> dict[str, Any]:
    manifest = source["manifest"]
    fields = (
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "dataset_checksums_sha256",
        "split_checksums_sha256",
        "cache_manifest_sha256",
    )
    result: dict[str, Any] = {}
    for field in fields:
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{source['role']} manifest field {field} is invalid")
        result[field] = value
    result["candidate_contract_sha256"] = _candidate_source_sha(source)
    benchmark_source = _require_mapping(
        manifest.get("benchmark_contract_source"),
        f"{source['role']}.manifest.benchmark_contract_source",
    )
    benchmark_sha = benchmark_source.get("sha256")
    if not isinstance(benchmark_sha, str) or len(benchmark_sha) != 64:
        raise ValueError(f"{source['role']} benchmark contract SHA is invalid")
    if source["config"].get("benchmark_contract_sha256") != benchmark_sha:
        raise ValueError(f"{source['role']} benchmark contract SHA binding mismatch")
    result["benchmark_contract_sha256"] = benchmark_sha
    audit_checks = _require_mapping(
        source["candidate_binding"].get("checks"),
        f"{source['role']}.candidate_binding.checks",
    )
    audit_to_manifest = {
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
        "cache_manifest_sha256": "cache_manifest_sha256",
        "base_benchmark_contract_sha256": "benchmark_contract_sha256",
    }
    for audit_name, fingerprint_name in audit_to_manifest.items():
        audit = _require_mapping(
            audit_checks.get(audit_name),
            f"{source['role']}.candidate_binding.checks.{audit_name}",
        )
        expected = audit.get("expected")
        actual = audit.get("actual")
        if expected != actual or actual != result[fingerprint_name]:
            raise ValueError(
                f"{source['role']} candidate binding check {audit_name!r} failed"
            )
    candidate_contract = source["candidate_contract"]
    parent = _require_mapping(
        candidate_contract.get("base_benchmark_contract"),
        f"{source['role']}.candidate_contract.base_benchmark_contract",
    )
    if parent.get("sha256") != result["benchmark_contract_sha256"]:
        raise ValueError(
            f"{source['role']} candidate contract base benchmark SHA mismatch"
        )
    development = _require_mapping(
        candidate_contract.get("development_dataset"),
        f"{source['role']}.candidate_contract.development_dataset",
    )
    contract_bindings = {
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "split_manifest_sha256": "split_manifest_sha256",
        "cache_manifest_sha256": "cache_manifest_sha256",
    }
    for contract_name, fingerprint_name in contract_bindings.items():
        if development.get(contract_name) != result[fingerprint_name]:
            raise ValueError(
                f"{source['role']} candidate contract {contract_name} mismatch"
            )
    return result


def _validate_execution_compatibility(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject any evaluator, environment, or policy drift between shards."""

    candidate_manifest = candidate["manifest"]
    comparator_manifest = comparator["manifest"]
    compared: dict[str, Any] = {}
    for field in (*EXECUTION_VERSION_FIELDS, *EXECUTION_POLICY_FIELDS):
        left = candidate_manifest.get(field)
        right = comparator_manifest.get(field)
        if left != right:
            raise ValueError(f"result shard execution mismatch: {field}")
        compared[field] = left
    for field in (
        "benchmark_runner_version",
        "device",
        "cache",
        "save_predictions",
        "test_splits",
        "train_validation_test_firewall",
        "candidate_contract",
    ):
        left = candidate["config"].get(field)
        right = comparator["config"].get(field)
        if left != right:
            raise ValueError(f"result shard benchmark-config mismatch: {field}")

    provenance = _require_mapping(
        candidate_manifest.get("provenance"),
        "candidate.run_manifest.provenance",
    )
    recorded_runner_sha = provenance.get("run_flowtwin_benchmark.py")
    current_runner_sha = sha256_file(RUNNER_SOURCE)
    if recorded_runner_sha != current_runner_sha:
        raise ValueError(
            "shard aggregation runner provenance does not match the current runner"
        )
    compared["aggregation_runner_source"] = {
        "path": str(RUNNER_SOURCE),
        "sha256": current_runner_sha,
    }
    return compared


def _profile_signature(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    metrics = _require_mapping(record.get("metrics"), f"{label}.metrics")
    by_profile = _require_mapping(metrics.get("by_profile"), f"{label}.by_profile")
    profile_splits = _require_mapping(
        metrics.get("profile_splits"), f"{label}.profile_splits"
    )
    if set(by_profile) != set(profile_splits) or not by_profile:
        raise ValueError(f"{label} profile metrics/splits are incomplete")
    signature: dict[str, Any] = {}
    for profile in sorted(by_profile):
        profile_metrics = _require_mapping(
            by_profile[profile], f"{label}.by_profile.{profile}"
        )
        rows = profile_metrics.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"{label} profile {profile} has invalid row count")
        split = profile_splits[profile]
        if not isinstance(split, str) or not split.startswith("test_"):
            raise ValueError(f"{label} profile {profile} has invalid split")
        for endpoint, (_scope, path, _direction) in ENDPOINTS.items():
            if _nested_number(profile_metrics, path) is None:
                raise ValueError(
                    f"{label} profile {profile} lacks finite endpoint {endpoint}"
                )
        signature[profile] = {"split": split, "rows": rows}
    return signature


def _validate_profile_completeness(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected: dict[str, Any] | None = None
    for index, record in enumerate(records):
        label = f"combined.runs[{index}]"
        signature = _profile_signature(record, label)
        if expected is None:
            expected = signature
        elif signature != expected:
            raise ValueError(
                f"{label} evaluated profile/split/row coverage differs from other runs"
            )
    assert expected is not None
    return expected


def _validate_and_combine(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    for source in (candidate, comparator):
        _validate_development_state(source)

    if candidate["candidate_contract"] != comparator["candidate_contract"]:
        raise ValueError("candidate contracts differ between result shards")
    if candidate["contract"] != comparator["contract"]:
        raise ValueError("benchmark contracts differ between result shards")
    execution_audit = _validate_execution_compatibility(candidate, comparator)
    candidate_fingerprint = _binding_fingerprint(candidate)
    comparator_fingerprint = _binding_fingerprint(comparator)
    if candidate_fingerprint != comparator_fingerprint:
        changed = sorted(
            key
            for key in set(candidate_fingerprint) | set(comparator_fingerprint)
            if candidate_fingerprint.get(key) != comparator_fingerprint.get(key)
        )
        raise ValueError(f"result shard artifact binding mismatch: {changed}")

    contract = candidate["candidate_contract"]
    candidate_id = contract.get("candidate_id")
    planned_raw = contract.get("planned_comparison_variants")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate contract candidate_id is invalid")
    if not isinstance(planned_raw, list) or len(planned_raw) != 4:
        raise ValueError("planned comparison must contain exactly four variants")
    planned = tuple(planned_raw)
    if any(not isinstance(value, str) or not value for value in planned):
        raise ValueError("planned comparison contains an invalid variant ID")
    if len(planned) != len(set(planned)):
        raise ValueError("candidate contract planned variants contain duplicates")
    if planned[0] != candidate_id:
        raise ValueError("candidate must be the first planned comparison variant")

    candidate_runs, candidate_seeds = _source_variant_runs(candidate)
    comparator_runs, comparator_seeds = _source_variant_runs(comparator)
    _verify_run_artifacts(candidate, candidate_runs)
    _verify_run_artifacts(comparator, comparator_runs)
    candidate_config_ids = _config_variant_ids(candidate)
    comparator_config_ids = _config_variant_ids(comparator)
    if candidate_config_ids != (candidate_id,) or set(candidate_seeds) != {candidate_id}:
        raise ValueError("candidate result directory must be candidate-only")
    expected_comparators = set(planned[1:])
    if set(comparator_config_ids) != expected_comparators or set(
        comparator_seeds
    ) != expected_comparators:
        raise ValueError("comparator shard does not exactly match planned comparators")
    if set(candidate_seeds) & set(comparator_seeds):
        raise ValueError("a variant is duplicated across candidate and comparator shards")
    union = set(candidate_seeds) | set(comparator_seeds)
    if union != set(planned):
        raise ValueError("variant union does not exactly match planned comparison")

    expected_budget = contract.get("training_budget")
    if not isinstance(expected_budget, dict):
        raise ValueError("candidate contract training_budget must be an object")
    candidate_budget = candidate["config"].get("budget")
    comparator_budget = comparator["config"].get("budget")
    if candidate_budget != expected_budget or comparator_budget != expected_budget:
        raise ValueError("result shard budget does not exactly match candidate contract")
    expected_seeds_raw = expected_budget.get("seeds")
    if (
        not isinstance(expected_seeds_raw, list)
        or not expected_seeds_raw
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in expected_seeds_raw)
        or len(expected_seeds_raw) != len(set(expected_seeds_raw))
    ):
        raise ValueError("candidate contract seeds must be unique integers")
    expected_seeds = set(expected_seeds_raw)
    for variant_id, actual_seeds in {**candidate_seeds, **comparator_seeds}.items():
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"variant {variant_id} does not contain the exact contracted seed set"
            )

    for source, source_runs in (
        (candidate, candidate_runs),
        (comparator, comparator_runs),
    ):
        configured = set(_config_variant_ids(source))
        observed = {_variant_id(run, "run") for run in source_runs}
        if configured != observed:
            raise ValueError(f"{source['role']} configured/result variants differ")

    combined = [*candidate_runs, *comparator_runs]
    order = {variant_id: index for index, variant_id in enumerate(planned)}
    combined.sort(key=lambda run: (order[_variant_id(run, "run")], int(run["seed"])))
    profile_signature = _validate_profile_completeness(combined)
    return (
        combined,
        candidate["contract"],
        planned,
        candidate_fingerprint,
        execution_audit,
        profile_signature,
    )


def _number_at(record: Mapping[str, Any], path: Sequence[str], label: str) -> float:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing compact-table metric {label}: {'.'.join(path)}")
        current = current[key]
    if current is None or isinstance(current, bool):
        raise ValueError(f"compact-table metric {label} is not numeric")
    try:
        number = float(current)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"compact-table metric {label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"compact-table metric {label} is not finite")
    return number


def _candidate_paired_comparisons(
    records: Sequence[Mapping[str, Any]],
    benchmark_contract: Mapping[str, Any],
    candidate_id: str,
    comparators: Sequence[str],
) -> dict[str, Any]:
    """Report candidate-minus-comparator effects at the profile unit."""

    values: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for record in records:
        variant_id = _variant_id(record, "combined run")
        seed = int(record["seed"])
        values.setdefault(variant_id, {name: {} for name in ENDPOINTS})
        metrics = record["metrics"]
        for profile, profile_metrics in metrics["by_profile"].items():
            split = metrics["profile_splits"][profile]
            for endpoint, (scope, path, _direction) in ENDPOINTS.items():
                if scope != "overall" and split != scope:
                    continue
                number = _nested_number(profile_metrics, path)
                if number is None:
                    raise ValueError(
                        f"non-finite profile endpoint {variant_id}/{profile}/{endpoint}"
                    )
                values[variant_id][endpoint].setdefault(profile, {})[seed] = number

    aggregation = _require_mapping(
        benchmark_contract.get("aggregation"), "benchmark_contract.aggregation"
    )
    repetitions = int(aggregation["bootstrap_repetitions"])
    base_seed = int(aggregation["bootstrap_seed"])
    result: dict[str, Any] = {}
    for comparator_id in comparators:
        result[comparator_id] = {}
        for endpoint, (_scope, _path, direction) in ENDPOINTS.items():
            comparison = _hierarchical_profile_bootstrap(
                values[candidate_id][endpoint],
                values[comparator_id][endpoint],
                repetitions=repetitions,
                seed=_stable_bootstrap_seed(
                    base_seed, candidate_id, comparator_id, endpoint
                ),
            )
            comparison["difference"] = "candidate_minus_comparator"
            comparison["candidate_id"] = candidate_id
            comparison["comparator_id"] = comparator_id
            comparison["direction"] = direction
            result[comparator_id][endpoint] = comparison
    return result


def _compact_table(
    records: Sequence[Mapping[str, Any]], planned: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in planned:
        variant_records = [
            record
            for record in records
            if _variant_id(record, "combined run") == variant_id
        ]
        if not variant_records:
            raise ValueError(f"no runs available for compact row {variant_id}")
        seeds = sorted(int(record["seed"]) for record in variant_records)
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "statistical_role": "descriptive_pooled_not_inferential",
            "technical_seed_count": len(seeds),
            "seeds": ";".join(map(str, seeds)),
        }
        for label, path in COMPACT_METRICS:
            values = [_number_at(record, path, label) for record in variant_records]
            row[label] = float(sum(values) / len(values))
        rows.append(row)
    return rows


def _write_compact_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "variant_id",
        "statistical_role",
        "technical_seed_count",
        "seeds",
        *(label for label, _path in COMPACT_METRICS),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes(source: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(source["root"])
    return {
        "role": source["role"],
        "path": str(root),
        "benchmark_results_sha256": sha256_file(root / "benchmark_results.json"),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
        "benchmark_config_sha256": sha256_file(root / "benchmark_config.json"),
        "candidate_contract_artifact_sha256": sha256_file(
            root / "candidate_contract.json"
        ),
    }


def _prepare_output(path: Path) -> tuple[Path, bool]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existed_empty = False
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        existed_empty = True
    temporary = Path(tempfile.mkdtemp(prefix=".flowtwin-v03-compare-", dir=output.parent))
    return temporary, existed_empty


def _commit_output(temporary: Path, output: Path, existed_empty: bool) -> None:
    resolved = Path(output).resolve()
    if existed_empty:
        resolved.rmdir()
    os.replace(temporary, resolved)


def compare(candidate_results: Path, comparator_results: Path, output: Path) -> Path:
    """Validate, aggregate, and atomically publish one v0.3 comparison."""

    temporary, existed_empty = _prepare_output(output)
    try:
        candidate = _load_source(candidate_results, "candidate")
        comparator = _load_source(comparator_results, "comparator")
        (
            records,
            benchmark_contract,
            planned,
            binding,
            execution_audit,
            profile_signature,
        ) = _validate_and_combine(candidate, comparator)
        aggregation = _aggregate_results(records, benchmark_contract)
        candidate_comparisons = _candidate_paired_comparisons(
            records,
            benchmark_contract,
            str(candidate["candidate_contract"]["candidate_id"]),
            planned[1:],
        )
        compact = _compact_table(records, planned)
        source_hashes = {
            "candidate": _source_hashes(candidate),
            "comparator": _source_hashes(comparator),
        }
        candidate_contract = candidate["candidate_contract"]
        results = {
            "comparison_version": COMPARISON_VERSION,
            "status": "post_hoc_development_only",
            "d2_test_seen_before_freeze": True,
            "eligible_for_confirmatory_claims": False,
            "candidate_id": candidate_contract["candidate_id"],
            "planned_comparison_variants": list(planned),
            "artifact_binding": binding,
            "budget": candidate_contract["training_budget"],
            "runs": records,
            "aggregation": aggregation,
            "candidate_paired_profile_comparisons": candidate_comparisons,
            "compact_table": compact,
            "compact_table_statistical_role": "descriptive_pooled_not_inferential",
            "execution_compatibility_audit": execution_audit,
            "complete_profile_signature": profile_signature,
            "claim_boundary": candidate_contract["claim_boundary"],
            "future_confirmatory_gate": candidate_contract[
                "future_confirmatory_gate"
            ],
            "source_artifacts": source_hashes,
            "merger_source": {
                "path": str(MERGER_SOURCE),
                "sha256": sha256_file(MERGER_SOURCE),
            },
        }
        write_json(temporary / "comparison_results.json", results)
        _write_compact_csv(temporary / "compact_table.csv", compact)
        manifest = {
            "comparison_version": COMPARISON_VERSION,
            "status": "post_hoc_development_only",
            "d2_test_seen_before_freeze": True,
            "eligible_for_confirmatory_claims": False,
            "candidate_contract_sha256": binding["candidate_contract_sha256"],
            "candidate_contract_semantic_sha256": sha256_json(candidate_contract),
            "benchmark_contract_sha256": binding["benchmark_contract_sha256"],
            "artifact_binding": binding,
            "exact_budget": candidate_contract["training_budget"],
            "exact_seeds": candidate_contract["training_budget"]["seeds"],
            "variant_order": list(planned),
            "source_artifacts": source_hashes,
            "merger_source": {
                "path": str(MERGER_SOURCE),
                "sha256": sha256_file(MERGER_SOURCE),
            },
            "aggregation_runner_source": execution_audit[
                "aggregation_runner_source"
            ],
            "execution_compatibility_audit": execution_audit,
            "complete_profile_signature": profile_signature,
            "artifacts": [*OUTPUT_ARTIFACTS, "checksums.sha256"],
            "claim_boundary": candidate_contract["claim_boundary"],
            "future_confirmatory_gate": candidate_contract[
                "future_confirmatory_gate"
            ],
        }
        write_json(temporary / "run_manifest.json", manifest)
        write_checksums(temporary, OUTPUT_ARTIFACTS)
        _commit_output(temporary, Path(output), existed_empty)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Path(output).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine matched FlowTwin v0.3 candidate/comparator shards"
    )
    parser.add_argument(
        "--candidate-results",
        "--candidate",
        dest="candidate_results",
        type=Path,
        required=True,
        help="checksummed candidate-only benchmark result directory",
    )
    parser.add_argument(
        "--comparator-results",
        "--comparators",
        dest="comparator_results",
        type=Path,
        required=True,
        help="checksummed matched comparator-shard result directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new atomic comparison directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = compare(args.candidate_results, args.comparator_results, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"FlowTwin v0.3 comparison complete: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
