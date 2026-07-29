#!/usr/bin/env python3
"""Re-evaluate the v0.3 alarm grid from a frozen TCN checkpoint.

This is a validation-only feasibility audit.  It reconstructs and strictly
reloads the already-trained, matched-budget TCN diagnostic checkpoint, opens
``validation_id`` exactly once, and applies the FlowTwin v0.3 calibration and
alarm-selection policy.  It never constructs a ``test_*`` dataset or reports
test performance.  Both a selected operating point and a
``no_operating_point`` result are published as permanent checksummed evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "FlowTwin audit dependencies are missing. Install requirements-ml.txt."
    ) from exc

from flowtwin_guard.cache import load_cache_manifest
from flowtwin_guard.data import FeatureStandardizer, load_target_contract
from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph
from ml_pipeline_common import (
    BASE_DIR,
    sha256_file,
    sha256_json,
    write_checksums,
    write_json,
)
from run_flowtwin_benchmark import (
    BENCHMARK_RUNNER_VERSION,
    DEFAULT_CONTRACT,
    DEFAULT_V03_CANDIDATE_CONTRACT,
    RuntimeContext,
    Variant,
    _collect,
    _fit_validation_calibration,
    _model_configs,
    _validate_v03_candidate_bindings,
    load_benchmark_contract,
    load_v03_candidate_contract,
)


ALARM_AUDIT_VERSION = "0.1.0"
EXPECTED_VARIANT = Variant("tcn", "neural_baseline", "tcn")

# Keep this list identical to the benchmark runner's recorded source_names.
# The audit verifies both the historic hashes in the raw run manifest and the
# files imported for checkpoint reconstruction today.
RUNTIME_SOURCE_NAMES = (
    "run_flowtwin_benchmark.py",
    "ml_pipeline_common.py",
    "reference_plant.py",
    "sensor_calibration.py",
    "fault_asset_contract.json",
    "ml_contract.json",
    "reference_pid.json",
    "sensor_catalog.json",
    "flowtwin_guard/graph.py",
    "flowtwin_guard/data.py",
    "flowtwin_guard/cache.py",
    "flowtwin_guard/model.py",
    "flowtwin_guard/hybrid.py",
    "flowtwin_guard/alarm.py",
    "flowtwin_guard/baselines.py",
    "flowtwin_guard/ablation.py",
    "flowtwin_guard/dspr.py",
    "flowtwin_guard/conformal.py",
    "flowtwin_guard/metrics.py",
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _strict_checksum_records(
    directory: Path, required_paths: Sequence[str]
) -> dict[str, str]:
    """Verify every declared checksum and require the audit's actual inputs."""

    root = Path(directory).resolve()
    checksum_path = root / "checksums.sha256"
    if not root.is_dir() or not checksum_path.is_file():
        raise ValueError(f"checksummed input directory is missing: {root}")
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
            raise ValueError(f"checksum path escapes input directory: {relative!r}")
        if relative in records:
            raise ValueError(f"duplicate checksum path in {checksum_path}: {relative}")
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"checksum target escapes input directory: {relative!r}"
            ) from exc
        if not target.is_file():
            raise ValueError(f"missing checksummed file in {root}: {relative}")
        actual = sha256_file(target)
        if actual != digest:
            raise ValueError(
                f"checksum mismatch for {target}: expected {digest}, got {actual}"
            )
        records[relative] = digest
    if not records:
        raise ValueError(f"checksums.sha256 is empty: {root}")
    missing = sorted(set(required_paths) - set(records))
    if missing:
        raise ValueError(f"required audit inputs are not checksummed in {root}: {missing}")
    return records


def _integrity_record(directory: Path, records: Mapping[str, str]) -> dict[str, Any]:
    root = Path(directory).resolve()
    return {
        "status": "pass",
        "path": str(root),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
        "declared_file_count": len(records),
    }


def _current_source_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in RUNTIME_SOURCE_NAMES:
        path = (BASE_DIR / name).resolve()
        try:
            path.relative_to(BASE_DIR.resolve())
        except ValueError as exc:  # pragma: no cover - constant contract guard
            raise ValueError(f"runtime source escapes BASE_DIR: {name}") from exc
        if not path.is_file():
            raise ValueError(f"required runtime source is missing: {name}")
        result[name] = sha256_file(path)
    return result


def _validate_raw_source_provenance(
    raw_manifest: Mapping[str, Any], benchmark_contract_path: Path
) -> dict[str, Any]:
    recorded = _require_mapping(
        raw_manifest.get("provenance"), "raw run_manifest.provenance"
    )
    expected_keys = {*RUNTIME_SOURCE_NAMES, "benchmark_contract"}
    if set(map(str, recorded)) != expected_keys:
        missing = sorted(expected_keys - set(map(str, recorded)))
        extra = sorted(set(map(str, recorded)) - expected_keys)
        raise ValueError(
            "raw run source provenance key set differs from the runner contract; "
            f"missing={missing}, extra={extra}"
        )
    current = _current_source_provenance()
    checks: dict[str, Any] = {}
    mismatches: list[str] = []
    for name in RUNTIME_SOURCE_NAMES:
        expected = str(recorded[name])
        actual = current[name]
        checks[name] = {"recorded": expected, "current": actual, "match": expected == actual}
        if expected != actual:
            mismatches.append(name)
    benchmark_sha = sha256_file(benchmark_contract_path)
    recorded_benchmark = str(recorded["benchmark_contract"])
    checks["benchmark_contract"] = {
        "recorded": recorded_benchmark,
        "current": benchmark_sha,
        "match": recorded_benchmark == benchmark_sha,
    }
    if recorded_benchmark != benchmark_sha:
        mismatches.append("benchmark_contract")
    manifest_contract = _require_mapping(
        raw_manifest.get("benchmark_contract_source"),
        "raw run_manifest.benchmark_contract_source",
    )
    if str(manifest_contract.get("sha256")) != benchmark_sha:
        mismatches.append("benchmark_contract_source")
    if mismatches:
        raise ValueError(
            "raw checkpoint source provenance does not match current reconstruction "
            f"sources: {sorted(set(mismatches))}"
        )
    return {
        "status": "pass",
        "exact_current_source_match": True,
        "source_count": len(RUNTIME_SOURCE_NAMES),
        "checks": checks,
        "recorded_provenance_sha256": sha256_json(dict(recorded)),
        "current_provenance_sha256": sha256_json(
            {**current, "benchmark_contract": benchmark_sha}
        ),
    }


def _prepare_output(path: Path) -> tuple[Path, Path, bool]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existed_empty = False
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        existed_empty = True
    temporary = Path(
        tempfile.mkdtemp(prefix=".flowtwin-v03-alarm-audit-", dir=output.parent)
    )
    return temporary, output, existed_empty


def _commit_output(temporary: Path, output: Path, existed_empty: bool) -> None:
    if existed_empty:
        output.rmdir()
    os.replace(temporary, output)


def _validate_raw_run(
    *,
    raw_result: Path,
    raw_manifest: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    raw_results: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    benchmark_contract_path: Path,
    dataset: Path,
    splits: Path,
    cache: Path,
    result_checksums: Mapping[str, str],
) -> tuple[dict[str, Any], Path, int]:
    """Validate the raw diagnostic shard and return its sole TCN run."""

    expected_budget = dict(candidate_contract["training_budget"])
    seeds = list(expected_budget.get("seeds", ()))
    if len(seeds) != 1 or isinstance(seeds[0], bool) or not isinstance(seeds[0], int):
        raise ValueError("candidate contract must bind exactly one integer seed")
    seed = int(seeds[0])
    if dict(raw_config.get("budget", {})) != expected_budget:
        raise ValueError("raw TCN budget is not the exact v0.3 candidate budget")
    if raw_config.get("device") != "cpu":
        raise ValueError("raw TCN checkpoint must come from the CPU reference path")
    if raw_config.get("candidate_contract") is not None:
        raise ValueError("raw diagnostic source must remain candidate-contract free")
    expected_variant = EXPECTED_VARIANT.to_dict()
    if raw_config.get("variants") != [expected_variant]:
        raise ValueError("raw result must configure exactly the matched TCN variant")
    planned = tuple(map(str, candidate_contract["planned_comparison_variants"]))
    if EXPECTED_VARIANT.variant_id not in planned:
        raise ValueError("candidate contract does not include TCN in its matched plan")

    runs = raw_results.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError("raw result must contain exactly one TCN run")
    run = dict(runs[0])
    expected_run_id = f"tcn__seed-{seed}"
    if (
        run.get("run_id") != expected_run_id
        or run.get("variant") != expected_variant
        or run.get("seed") != seed
    ):
        raise ValueError("raw result run_id/variant/seed does not match v0.3")
    run_record_path = raw_result / "runs" / expected_run_id / "run_record.json"
    run_record = _json_object(run_record_path)
    if run_record != run:
        raise ValueError("benchmark_results run and checksummed run_record differ")
    checkpoint = raw_result / "runs" / expected_run_id / "model.pt"
    required_result_paths = {
        "benchmark_config.json",
        "benchmark_results.json",
        "run_manifest.json",
        "standardizer.json",
        str(run_record_path.relative_to(raw_result)),
        str(checkpoint.relative_to(raw_result)),
    }
    missing = sorted(required_result_paths - set(result_checksums))
    if missing:
        raise ValueError(f"raw result omits required checksummed artifacts: {missing}")
    checkpoint_sha = sha256_file(checkpoint)
    if (
        str(run.get("checkpoint_sha256")) != checkpoint_sha
        or result_checksums[str(checkpoint.relative_to(raw_result))] != checkpoint_sha
    ):
        raise ValueError("raw checkpoint SHA does not match its result records")

    input_hashes = {
        "dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
        "split_manifest_sha256": sha256_file(splits / "split_manifest.json"),
        "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
        "dataset_checksums_sha256": sha256_file(dataset / "checksums.sha256"),
        "split_checksums_sha256": sha256_file(splits / "checksums.sha256"),
    }
    mismatches = [
        name
        for name, actual in input_hashes.items()
        if str(raw_manifest.get(name)) != actual
    ]
    if mismatches:
        raise ValueError(f"raw run input provenance mismatch: {mismatches}")
    if raw_manifest.get("benchmark_runner_version") != BENCHMARK_RUNNER_VERSION:
        raise ValueError("raw run benchmark runner version mismatch")
    if raw_config.get("benchmark_runner_version") != BENCHMARK_RUNNER_VERSION:
        raise ValueError("raw benchmark config runner version mismatch")
    split_policy = _require_mapping(
        raw_manifest.get("split_policy"), "raw run_manifest.split_policy"
    )
    if split_policy.get("calibration") != ["validation_id"]:
        raise ValueError("raw run calibration provenance is not validation_id-only")
    benchmark_sha = sha256_file(benchmark_contract_path)
    if str(raw_config.get("benchmark_contract_sha256")) != benchmark_sha:
        raise ValueError("raw benchmark config contract hash mismatch")
    raw_environment = _require_mapping(
        raw_manifest.get("environment"), "raw run_manifest.environment"
    )
    expected_environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
    }
    environment_mismatches = [
        name
        for name, expected in expected_environment.items()
        if raw_environment.get(name) != expected
    ]
    if environment_mismatches:
        raise ValueError(
            "raw checkpoint environment differs from the audit environment: "
            f"{environment_mismatches}"
        )
    return run, checkpoint, seed


def _reconstruct_and_reload(
    *,
    checkpoint_path: Path,
    raw_run: Mapping[str, Any],
    seed: int,
    context: RuntimeContext,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, serialized, protocol = _model_configs(context, EXPECTED_VARIANT)
    try:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load raw TCN checkpoint safely: {exc}") from exc
    if not isinstance(saved, dict) or set(saved) != {
        "benchmark_runner_version",
        "variant",
        "seed",
        "serialized_model",
        "state_dict",
    }:
        raise ValueError("raw checkpoint has an unexpected top-level schema")
    if saved["benchmark_runner_version"] != BENCHMARK_RUNNER_VERSION:
        raise ValueError("checkpoint runner version mismatch")
    if saved["variant"] != EXPECTED_VARIANT.to_dict() or saved["seed"] != seed:
        raise ValueError("checkpoint variant/seed mismatch")
    if saved["serialized_model"] != serialized or raw_run.get("model") != serialized:
        raise ValueError("checkpoint model configuration cannot be strictly reconstructed")
    if raw_run.get("training_protocol") != protocol["training"]:
        raise ValueError("raw training protocol differs from reconstructed TCN")
    if raw_run.get("evaluation_protocol") != protocol["evaluation"]:
        raise ValueError("raw evaluation protocol differs from reconstructed TCN")
    try:
        model.load_state_dict(saved["state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"strict checkpoint state reload failed: {exc}") from exc
    if hasattr(model, "set_observer_trainable"):
        model.set_observer_trainable(False)
    model.eval()
    return model, protocol


def audit(
    *,
    raw_result: Path,
    dataset: Path,
    splits: Path,
    cache: Path,
    output: Path,
    benchmark_contract: Path = DEFAULT_CONTRACT,
    candidate_contract: Path = DEFAULT_V03_CANDIDATE_CONTRACT,
) -> Path:
    """Create an atomic, checksummed validation-only alarm feasibility audit."""

    raw_result = Path(raw_result).resolve()
    dataset = Path(dataset).resolve()
    splits = Path(splits).resolve()
    cache = Path(cache).resolve()
    benchmark_contract = Path(benchmark_contract).resolve()
    candidate_contract = Path(candidate_contract).resolve()

    candidate = load_v03_candidate_contract(candidate_contract)
    # Loading the base contract catches registry or protocol corruption even
    # though no benchmark training or test evaluation occurs in this tool.
    load_benchmark_contract(benchmark_contract)
    seed = int(candidate["training_budget"]["seeds"][0])
    raw_run_id = f"tcn__seed-{seed}"
    result_records = _strict_checksum_records(
        raw_result,
        (
            "benchmark_config.json",
            "benchmark_results.json",
            "run_manifest.json",
            "split_domain_audit.json",
            "standardizer.json",
            f"runs/{raw_run_id}/model.pt",
            f"runs/{raw_run_id}/run_record.json",
        ),
    )
    dataset_records = _strict_checksum_records(dataset, ("dataset_manifest.json",))
    split_records = _strict_checksum_records(splits, ("split_manifest.json",))
    cache_records = _strict_checksum_records(cache, ("cache_manifest.json",))
    input_integrity = {
        "raw_result": _integrity_record(raw_result, result_records),
        "dataset": _integrity_record(dataset, dataset_records),
        "splits": _integrity_record(splits, split_records),
        "cache": _integrity_record(cache, cache_records),
    }

    raw_manifest = _json_object(raw_result / "run_manifest.json")
    raw_config = _json_object(raw_result / "benchmark_config.json")
    raw_results = _json_object(raw_result / "benchmark_results.json")
    split_domain_audit = _json_object(raw_result / "split_domain_audit.json")
    if raw_results.get("split_domain_audit") != split_domain_audit:
        raise ValueError(
            "raw benchmark results and split_domain_audit artifact differ"
        )
    source_audit = _validate_raw_source_provenance(
        raw_manifest, benchmark_contract
    )
    raw_run, checkpoint_path, seed = _validate_raw_run(
        raw_result=raw_result,
        raw_manifest=raw_manifest,
        raw_config=raw_config,
        raw_results=raw_results,
        candidate_contract=candidate,
        benchmark_contract_path=benchmark_contract,
        dataset=dataset,
        splits=splits,
        cache=cache,
        result_checksums=result_records,
    )
    binding = _validate_v03_candidate_bindings(
        candidate,
        candidate_contract_path=candidate_contract,
        benchmark_contract_path=benchmark_contract,
        dataset=dataset,
        splits=splits,
        cache=cache,
        budget=dict(candidate["training_budget"]),
    )

    graph = build_process_graph(
        feature_set=str(candidate["training_budget"]["feature_set"])
    )
    targets = load_target_contract(graph=graph)
    cache_manifest = load_cache_manifest(cache, dataset, splits, graph, targets)
    raw_standardizer = _json_object(raw_result / "standardizer.json")
    if raw_standardizer != cache_manifest.get("standardizer"):
        raise ValueError("raw result and bound cache standardizers differ")
    standardizer = FeatureStandardizer.from_dict(raw_standardizer)
    context = RuntimeContext(
        dataset=dataset,
        splits=splits,
        cache=cache,
        cache_manifest=cache_manifest,
        graph=graph,
        targets=targets,
        standardizer=standardizer,
        budget=dict(candidate["training_budget"]),
        device=torch.device("cpu"),
        # Reconstruct the raw checkpoint exactly; the alarm policy is attached
        # only after strict reload below.
        candidate_contract=None,
    )
    model, protocol = _reconstruct_and_reload(
        checkpoint_path=checkpoint_path,
        raw_run=raw_run,
        seed=seed,
        context=context,
    )

    # Firewall: this is the sole data-collection call in the audit.  There is
    # deliberately no split discovery and no code path that constructs test.
    validation, validation_seconds_per_row = _collect(
        model, context, ("validation_id",)
    )
    if validation.get("split_vocabulary") != ["validation_id"]:
        raise ValueError("validation collection returned a non-validation split")
    evaluation_options = dict(protocol["evaluation"])
    evaluation_options["operational_alarm"] = dict(candidate["alarm_policy"])
    calibration, _calibrator = _fit_validation_calibration(
        validation, context, evaluation_options
    )
    alarm = _require_mapping(
        calibration.get("operational_alarm"), "operational alarm result"
    )
    alarm_audit = _require_mapping(alarm.get("audit"), "operational alarm audit")
    candidate_rows = alarm_audit.get("candidates")
    if not isinstance(candidate_rows, list):
        raise ValueError("operational alarm audit lacks its full candidate grid")
    reason_counts = Counter(
        str(reason)
        for row in candidate_rows
        if isinstance(row, Mapping)
        for reason in row.get("rejection_reasons", ())
    )
    status = str(alarm.get("status"))
    if status not in {"selected", "no_operating_point"}:
        raise ValueError(f"unexpected operational alarm status: {status!r}")
    feasible_count = int(alarm_audit.get("feasible_candidate_count", -1))
    if feasible_count < 0:
        raise ValueError("operational alarm feasible-candidate count is invalid")
    rejection_reasons = list(alarm_audit.get("no_operating_point_reasons", ()))

    temporary, final_output, existed_empty = _prepare_output(output)
    try:
        validation_access = {
            "alarm_audit_version": ALARM_AUDIT_VERSION,
            "status": "pass",
            "evidence_status": "post_hoc_development_only",
            "opened_splits": ["validation_id"],
            "collection_calls": 1,
            "validation_rows": int(len(validation["class_target"])),
            "test_iterator_constructed": False,
            "test_rows_seen": 0,
            "fit_reported_test_rows_seen": int(
                calibration.get("test_rows_seen_during_fit", -1)
            ),
            "validation_inference_rows_per_s": (
                1.0 / validation_seconds_per_row
                if validation_seconds_per_row > 0.0
                else None
            ),
            "raw_checkpoint_source_run_had_prior_test_evaluation": True,
            "prior_test_results_used_for_alarm_fit_or_selection": False,
            "prior_test_metrics_parsed_for_lineage_validation": True,
            "test_prediction_arrays_decoded": False,
            "integrity_hashing_reads_declared_input_bytes": True,
            "test_rows_decoded_for_model_or_metrics": 0,
            "claim_boundary": candidate["claim_boundary"],
        }
        if validation_access["fit_reported_test_rows_seen"] != 0:
            raise ValueError("calibration fit reports non-zero test-row access")

        feasibility = {
            "alarm_audit_version": ALARM_AUDIT_VERSION,
            "artifact_status": "complete",
            "validation_gate_status": status,
            "evidence_status": "post_hoc_development_only",
            "eligible_for_confirmatory_claims": False,
            "variant": EXPECTED_VARIANT.to_dict(),
            "seed": seed,
            "budget": dict(candidate["training_budget"]),
            "alarm_fit": {
                "status": status,
                "config": alarm.get("config"),
                "candidate_count": int(alarm_audit.get("candidate_count", -1)),
                "feasible_candidate_count": feasible_count,
                "no_operating_point_reasons": rejection_reasons,
                "candidate_rejection_reason_counts": dict(
                    sorted(reason_counts.items())
                ),
                # Full grid, per-profile constraints, and deterministic ties.
                "full_audit": dict(alarm_audit),
            },
            "claim_boundary": candidate["claim_boundary"],
        }
        raw_environment = dict(
            _require_mapping(
                raw_manifest.get("environment"), "raw run_manifest.environment"
            )
        )
        audit_environment = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        profile_binding = {
            "source_artifact_sha256": sha256_file(
                raw_result / "split_domain_audit.json"
            ),
            "canonical_content_sha256": sha256_json(split_domain_audit),
            "status": split_domain_audit.get("status"),
            "episodes": split_domain_audit.get("episodes"),
            "profiles": split_domain_audit.get("profiles"),
            "counterfactual_groups": split_domain_audit.get(
                "counterfactual_groups"
            ),
            "domain_episode_counts": split_domain_audit.get(
                "domain_episode_counts"
            ),
            "observed_ood_domains": split_domain_audit.get(
                "observed_ood_domains"
            ),
            "domain_contract_verified": split_domain_audit.get(
                "domain_contract_verified"
            ),
            "profile_table_domain_verified": split_domain_audit.get(
                "profile_table_domain_verified"
            ),
            "profile_parameter_ranges_verified": split_domain_audit.get(
                "profile_parameter_ranges_verified"
            ),
        }
        checkpoint_lineage = {
            "alarm_audit_version": ALARM_AUDIT_VERSION,
            "status": "pass",
            "variant": EXPECTED_VARIANT.to_dict(),
            "seed": seed,
            "budget": dict(candidate["training_budget"]),
            "candidate_contract_sha256": sha256_file(candidate_contract),
            "candidate_binding_audit": binding,
            "input_integrity": input_integrity,
            "raw_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "strict_reload": "pass",
                "source_run_id": raw_run["run_id"],
                "source_result_checksums_sha256": sha256_file(
                    raw_result / "checksums.sha256"
                ),
                "source_run_had_prior_test_evaluation": True,
                "prior_test_results_used_for_alarm_fit_or_selection": False,
                "prior_test_metrics_parsed_for_lineage_validation": True,
                "test_prediction_arrays_decoded": False,
            },
            "source_artifact_hashes": {
                "benchmark_config_sha256": sha256_file(
                    raw_result / "benchmark_config.json"
                ),
                "benchmark_results_sha256": sha256_file(
                    raw_result / "benchmark_results.json"
                ),
                "run_manifest_sha256": sha256_file(
                    raw_result / "run_manifest.json"
                ),
                "split_domain_audit_sha256": sha256_file(
                    raw_result / "split_domain_audit.json"
                ),
                "standardizer_sha256": sha256_file(
                    raw_result / "standardizer.json"
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            },
            "profile_domain_binding": profile_binding,
            "environment_binding": {
                "status": "pass",
                "raw_checkpoint_environment": raw_environment,
                "audit_environment": audit_environment,
            },
            "source_provenance_audit": source_audit,
            "claim_boundary": candidate["claim_boundary"],
        }
        write_json(temporary / "alarm_feasibility.json", feasibility)
        write_json(temporary / "checkpoint_lineage.json", checkpoint_lineage)
        write_json(temporary / "validation_access_audit.json", validation_access)
        write_json(temporary / "calibration.json", calibration)
        current_sources = _current_source_provenance()
        manifest = {
            "alarm_audit_version": ALARM_AUDIT_VERSION,
            "artifact_status": "complete",
            "validation_gate_status": status,
            "variant_id": EXPECTED_VARIANT.variant_id,
            "seed": seed,
            "artifact_hashes": {
                name: sha256_file(temporary / name)
                for name in (
                    "alarm_feasibility.json",
                    "checkpoint_lineage.json",
                    "validation_access_audit.json",
                    "calibration.json",
                )
            },
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "benchmark_contract_source": {
                "path": str(benchmark_contract),
                "sha256": sha256_file(benchmark_contract),
            },
            "candidate_contract_source": {
                "path": str(candidate_contract),
                "sha256": sha256_file(candidate_contract),
            },
            "dataset_manifest_sha256": sha256_file(
                dataset / "dataset_manifest.json"
            ),
            "split_manifest_sha256": sha256_file(splits / "split_manifest.json"),
            "dataset_checksums_sha256": sha256_file(
                dataset / "checksums.sha256"
            ),
            "split_checksums_sha256": sha256_file(splits / "checksums.sha256"),
            "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
            "raw_result_checksums_sha256": sha256_file(
                raw_result / "checksums.sha256"
            ),
            "environment": audit_environment,
            "raw_checkpoint_environment": raw_environment,
            "profile_domain_binding": profile_binding,
            "source_provenance": {
                **current_sources,
                "audit_flowtwin_v03_alarm.py": sha256_file(Path(__file__)),
            },
            "source_provenance_audit": source_audit,
            "raw_training_source_provenance_sha256": source_audit[
                "recorded_provenance_sha256"
            ],
            "split_policy": {
                "calibration": ["validation_id"],
                "evaluation": [],
                "test_iterator_constructed": False,
                "test_rows_seen": 0,
            },
            "input_integrity": input_integrity,
            "development_disclosure": {
                "d2_test_seen_before_candidate_freeze": True,
                "raw_checkpoint_source_run_had_prior_test_evaluation": True,
                "prior_test_results_used_for_alarm_fit_or_selection": False,
                "prior_test_metrics_parsed_for_lineage_validation": True,
                "test_prediction_arrays_decoded": False,
                "integrity_hashing_reads_declared_input_bytes": True,
                "eligible_for_confirmatory_claims": False,
            },
            "claim_boundary": candidate["claim_boundary"],
        }
        write_json(temporary / "run_manifest.json", manifest)
        write_checksums(
            temporary,
            (
                "alarm_feasibility.json",
                "checkpoint_lineage.json",
                "validation_access_audit.json",
                "calibration.json",
                "run_manifest.json",
            ),
        )
        _commit_output(temporary, final_output, existed_empty)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        default=DEFAULT_V03_CANDIDATE_CONTRACT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = audit(
            raw_result=args.raw_result,
            dataset=args.dataset,
            splits=args.splits,
            cache=args.cache,
            output=args.output,
            benchmark_contract=args.benchmark_contract,
            candidate_contract=args.candidate_contract,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    result = _json_object(output / "alarm_feasibility.json")
    fit = _require_mapping(result.get("alarm_fit"), "alarm_feasibility.alarm_fit")
    print(
        f"FlowTwin v0.3 alarm audit complete: {output}\n"
        f"status={result['validation_gate_status']}, "
        f"feasible={fit['feasible_candidate_count']}/{fit['candidate_count']}, "
        "test_rows_seen=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
