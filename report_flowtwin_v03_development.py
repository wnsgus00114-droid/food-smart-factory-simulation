#!/usr/bin/env python3
"""Publish the disclosed FlowTwin v0.3 opened-D2 development report.

The operational contract failed closed for TCN.  This reporter therefore keeps
four-model diagnostic evidence separate from validation feasibility and from
the conditional operational test results of models that passed validation.
It never substitutes a zero or a fallback threshold for TCN.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from compare_flowtwin_v03 import (
    EXECUTION_POLICY_FIELDS,
    EXECUTION_VERSION_FIELDS,
    _binding_fingerprint,
    _json_object,
    _load_source,
    _profile_signature,
    _source_variant_runs,
    _validate_development_state,
    _validate_execution_compatibility,
    _verify_run_artifacts,
)
from ml_pipeline_common import (
    read_checksums,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
)
from run_flowtwin_benchmark import (
    _hierarchical_profile_bootstrap,
    _nested_number,
    _stable_bootstrap_seed,
)


REPORT_VERSION = "0.1.1"
VALIDATION_HELPER_SOURCE = Path(__file__).resolve().parent / "compare_flowtwin_v03.py"
VARIANT_ORDER = (
    "flowtwin_hybrid_v03_dev",
    "flowtwin_guard",
    "tcn",
    "dspr_diagnostic_adaptation",
)
DIAGNOSTIC_ENDPOINTS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "id_class_macro_f1": (
        "test_id",
        ("class_diagnosis", "macro_f1"),
        "higher_is_better",
    ),
    "validation_fitted_row_event_f1": (
        "overall",
        ("event_detection", "event_f1"),
        "higher_is_better",
    ),
    "validation_fitted_row_event_recall": (
        "overall",
        ("event_detection", "event_recall"),
        "higher_is_better",
    ),
    "validation_fitted_row_false_alarm_onsets_per_negative_hour": (
        "overall",
        ("false_alarms", "false_alarm_onsets_per_negative_hour"),
        "lower_is_better",
    ),
    "validation_fitted_row_unsafe_forward_l_before_detection": (
        "overall",
        ("unsafe_forward_l_before_first_post_effect_alarm",),
        "lower_is_better",
    ),
    "conformal_coverage": (
        "overall",
        ("conformal", "empirical_coverage"),
        "target_is_one_minus_alpha",
    ),
    "conformal_mean_prediction_set_size": (
        "overall",
        ("conformal", "mean_prediction_set_size"),
        "lower_at_valid_coverage_is_better",
    ),
}
OPERATIONAL_ENDPOINTS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "operational_event_f1": (
        "overall",
        ("operational_event_detection", "event_f1"),
        "higher_is_better",
    ),
    "operational_event_recall": (
        "overall",
        ("operational_event_detection", "event_recall"),
        "higher_is_better",
    ),
    "operational_false_alarm_onsets_per_negative_hour": (
        "overall",
        ("operational_false_alarms", "false_alarm_onsets_per_negative_hour"),
        "lower_is_better",
    ),
    "operational_unsafe_forward_l_before_detection": (
        "overall",
        ("operational_unsafe_forward_l_before_first_post_effect_alarm",),
        "lower_is_better",
    ),
}
OUTPUT_FILES = (
    "development_report.json",
    "diagnostic_table.csv",
    "operational_feasibility.csv",
    "conditional_operational_table.csv",
    "run_manifest.json",
)


def _only_run(source: Mapping[str, Any], expected_variant: str) -> dict[str, Any]:
    runs, seeds = _source_variant_runs(source)
    if len(runs) != 1 or set(seeds) != {expected_variant}:
        raise ValueError(
            f"{source['role']} must contain exactly one {expected_variant} run"
        )
    _verify_run_artifacts(source, runs)
    return runs[0]


def _load_raw_tcn(path: Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures = verify_checksums(root)
    if failures:
        raise ValueError(f"TCN raw result checksum failure: {failures}")
    required = (
        "benchmark_config.json",
        "benchmark_contract.json",
        "benchmark_results.json",
        "run_manifest.json",
    )
    checksums = read_checksums(root)
    missing = [name for name in required if name not in checksums]
    if missing:
        raise ValueError(f"TCN raw result lacks checksummed artifacts: {missing}")
    return {
        "role": "tcn_raw_diagnostic",
        "root": root,
        "checksums": checksums,
        "config": _json_object(root / "benchmark_config.json"),
        "contract": _json_object(root / "benchmark_contract.json"),
        "results": _json_object(root / "benchmark_results.json"),
        "manifest": _json_object(root / "run_manifest.json"),
    }


def _common_execution_without_candidate(
    candidate: Mapping[str, Any], tcn: Mapping[str, Any]
) -> None:
    for field in EXECUTION_VERSION_FIELDS:
        if candidate["manifest"].get(field) != tcn["manifest"].get(field):
            raise ValueError(f"TCN execution version mismatch: {field}")
    for field in EXECUTION_POLICY_FIELDS:
        left = candidate["manifest"].get(field)
        right = tcn["manifest"].get(field)
        if field == "provenance":
            left = dict(left or {})
            right = dict(right or {})
            left.pop("candidate_contract", None)
            right.pop("candidate_contract", None)
        if left != right:
            raise ValueError(f"TCN execution policy mismatch: {field}")
    for field in (
        "benchmark_runner_version",
        "device",
        "cache",
        "save_predictions",
        "test_splits",
        "train_validation_test_firewall",
    ):
        if candidate["config"].get(field) != tcn["config"].get(field):
            raise ValueError(f"TCN benchmark config mismatch: {field}")


def _input_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "dataset_checksums_sha256",
            "split_checksums_sha256",
            "cache_manifest_sha256",
        )
    }


def _diagnostic_profile_signature(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected: dict[str, Any] | None = None
    for record in records:
        metrics = record["metrics"]
        profiles = metrics.get("by_profile")
        splits = metrics.get("profile_splits")
        if not isinstance(profiles, Mapping) or not isinstance(splits, Mapping):
            raise ValueError("profile metrics are missing")
        if set(profiles) != set(splits) or not profiles:
            raise ValueError("profile metrics/splits are incomplete")
        signature: dict[str, Any] = {}
        for profile in sorted(profiles):
            scope = profiles[profile]
            rows = scope.get("rows")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise ValueError(f"invalid profile row count: {profile}")
            for endpoint, (_split, path, _direction) in DIAGNOSTIC_ENDPOINTS.items():
                if _nested_number(scope, path) is None:
                    raise ValueError(f"missing profile endpoint {profile}/{endpoint}")
            signature[profile] = {"split": splits[profile], "rows": rows}
        if expected is None:
            expected = signature
        elif signature != expected:
            raise ValueError("four-model diagnostic profile coverage differs")
    assert expected is not None
    return expected


def _metric(scope: Mapping[str, Any], path: Sequence[str]) -> float:
    value = _nested_number(scope, path)
    if value is None:
        raise ValueError(f"missing finite metric: {'.'.join(path)}")
    return value


def _diagnostic_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    overall = metrics["overall"]
    by_split = metrics["by_split"]
    efficiency = metrics["efficiency"]
    return {
        "variant_id": record["variant"]["variant_id"],
        "statistical_role": "descriptive_pooled_not_inferential",
        "overall_class_macro_f1": _metric(overall, ("class_diagnosis", "macro_f1")),
        "id_class_macro_f1": _metric(
            by_split["test_id"], ("class_diagnosis", "macro_f1")
        ),
        "ood_class_macro_f1": _metric(
            by_split["test_ood_profile"], ("class_diagnosis", "macro_f1")
        ),
        "validation_fitted_row_event_f1": _metric(
            overall, ("event_detection", "event_f1")
        ),
        "validation_fitted_row_event_recall": _metric(
            overall, ("event_detection", "event_recall")
        ),
        "validation_fitted_row_false_alarm_onsets_per_negative_hour": _metric(
            overall, ("false_alarms", "false_alarm_onsets_per_negative_hour")
        ),
        "validation_fitted_row_unsafe_forward_l_before_detection": _metric(
            overall, ("unsafe_forward_l_before_first_post_effect_alarm",)
        ),
        "ood_auroc": _metric(metrics["ood_detection"], ("auroc",)),
        "ood_aupr": _metric(metrics["ood_detection"], ("aupr",)),
        "ood_fpr95": _metric(metrics["ood_detection"], ("fpr95",)),
        "conformal_coverage": _metric(overall, ("conformal", "empirical_coverage")),
        "conformal_mean_prediction_set_size": _metric(
            overall, ("conformal", "mean_prediction_set_size")
        ),
        "conformal_singleton_fraction": _metric(
            overall, ("conformal", "singleton_fraction")
        ),
        "parameter_count": int(efficiency["parameter_count"]),
        "training_wall_time_s_descriptive_only": float(
            efficiency["training_wall_time_s"]
        ),
        "test_inference_rows_per_s_descriptive_only": float(
            efficiency["test_inference_rows_per_s"]
        ),
        "timing_comparable_across_variants": False,
    }


def _load_calibration(source: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return _json_object(
        Path(source["root"]) / "runs" / record["run_id"] / "calibration.json"
    )


def _selected_feasibility(
    source: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    alarm = _load_calibration(source, record).get("operational_alarm")
    if not isinstance(alarm, Mapping) or alarm.get("status") != "selected":
        raise ValueError(f"{source['role']} lacks a selected operational alarm")
    audit = alarm["audit"]
    selected = audit["selected_validation_metrics"]
    return {
        "variant_id": record["variant"]["variant_id"],
        "validation_gate_status": "selected",
        "candidate_count": int(audit["candidate_count"]),
        "feasible_candidate_count": int(audit["feasible_candidate_count"]),
        "selected_config": alarm["config"],
        "validation_event_f1": _metric(selected, ("event_metrics", "event_f1")),
        "validation_event_recall": _metric(
            selected, ("event_metrics", "event_recall")
        ),
        "validation_minimum_profile_event_recall": _metric(
            selected,
            ("profile_event_recall_audit", "minimum_profile_event_recall"),
        ),
        "validation_worst_profile_false_alarm_onsets_per_negative_hour": _metric(
            selected,
            (
                "profile_false_alarm_audit",
                "maximum_profile_false_alarm_onsets_per_negative_hour",
            ),
        ),
        "operational_test_evaluated": True,
        "operational_test_metrics": "reported_conditionally_on_validation_feasibility",
    }


def _validate_tcn_audit(
    audit_root: Path,
    tcn: Mapping[str, Any],
    tcn_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(audit_root).resolve()
    failures = verify_checksums(root)
    if failures:
        raise ValueError(f"TCN alarm audit checksum failure: {failures}")
    feasibility = _json_object(root / "alarm_feasibility.json")
    access = _json_object(root / "validation_access_audit.json")
    lineage = _json_object(root / "checkpoint_lineage.json")
    alarm = feasibility.get("alarm_fit")
    if (
        feasibility.get("artifact_status") != "complete"
        or feasibility.get("validation_gate_status") != "no_operating_point"
        or not isinstance(alarm, Mapping)
        or alarm.get("status") != "no_operating_point"
        or alarm.get("config") is not None
        or alarm.get("feasible_candidate_count") != 0
    ):
        raise ValueError("TCN audit is not the required no-operating-point result")
    if (
        access.get("test_iterator_constructed") is not False
        or access.get("test_rows_seen") != 0
        or access.get("opened_splits") != ["validation_id"]
    ):
        raise ValueError("TCN alarm audit accessed a test split")
    checkpoint = Path(tcn["root"]) / "runs" / tcn_record["run_id"] / "model.pt"
    raw_checkpoint = lineage.get("raw_checkpoint", {})
    if raw_checkpoint.get("sha256") != sha256_file(checkpoint):
        raise ValueError("TCN audit checkpoint lineage mismatch")
    audit_data = alarm["full_audit"]
    row = {
        "variant_id": "tcn",
        "validation_gate_status": "no_operating_point",
        "candidate_count": int(audit_data["candidate_count"]),
        "feasible_candidate_count": 0,
        "selected_config": None,
        "validation_event_f1": None,
        "validation_event_recall": None,
        "validation_minimum_profile_event_recall": None,
        "validation_worst_profile_false_alarm_onsets_per_negative_hour": None,
        "no_operating_point_reasons": audit_data["no_operating_point_reasons"],
        "operational_test_evaluated": False,
        "operational_test_metrics": None,
    }
    return row, {
        "path": str(root),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
    }


def _conditional_operational_row(record: Mapping[str, Any]) -> dict[str, Any]:
    overall = record["metrics"]["overall"]
    profiles = record["metrics"]["by_profile"]
    return {
        "variant_id": record["variant"]["variant_id"],
        "statistical_role": "conditional_on_validation_feasibility",
        "event_f1": _metric(overall, ("operational_event_detection", "event_f1")),
        "event_recall": _metric(
            overall, ("operational_event_detection", "event_recall")
        ),
        "pooled_false_alarm_onsets_per_negative_hour": _metric(
            overall,
            ("operational_false_alarms", "false_alarm_onsets_per_negative_hour"),
        ),
        "maximum_profile_false_alarm_onsets_per_negative_hour": max(
            _metric(
                value,
                (
                    "operational_false_alarms",
                    "false_alarm_onsets_per_negative_hour",
                ),
            )
            for value in profiles.values()
        ),
        "unsafe_forward_l_before_detection": _metric(
            overall,
            ("operational_unsafe_forward_l_before_first_post_effect_alarm",),
        ),
    }


def _paired_comparisons(
    records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, tuple[str, tuple[str, ...], str]],
    comparators: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for record in records:
        variant = str(record["variant"]["variant_id"])
        seed = int(record["seed"])
        values.setdefault(variant, {name: {} for name in endpoints})
        metrics = record["metrics"]
        for profile, scope in metrics["by_profile"].items():
            split = metrics["profile_splits"][profile]
            for endpoint, (required_split, path, _direction) in endpoints.items():
                if required_split != "overall" and split != required_split:
                    continue
                number = _nested_number(scope, path)
                if number is None:
                    raise ValueError(f"missing paired endpoint {variant}/{profile}/{endpoint}")
                values[variant][endpoint].setdefault(profile, {})[seed] = number
    aggregation = contract["aggregation"]
    result: dict[str, Any] = {}
    for comparator in comparators:
        result[comparator] = {}
        for endpoint, (_split, _path, direction) in endpoints.items():
            item = _hierarchical_profile_bootstrap(
                values[VARIANT_ORDER[0]][endpoint],
                values[comparator][endpoint],
                repetitions=int(aggregation["bootstrap_repetitions"]),
                seed=_stable_bootstrap_seed(
                    int(aggregation["bootstrap_seed"]),
                    REPORT_VERSION,
                    comparator,
                    endpoint,
                ),
            )
            item.update(
                {
                    "difference": "candidate_minus_comparator",
                    "direction": direction,
                    "candidate_id": VARIANT_ORDER[0],
                    "comparator_id": comparator,
                }
            )
            result[comparator][endpoint] = item
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _atomic_output(path: Path) -> tuple[Path, bool]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existed_empty = False
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        existed_empty = True
    return (
        Path(tempfile.mkdtemp(prefix=".flowtwin-v03-report-", dir=output.parent)),
        existed_empty,
    )


def build_report(
    *,
    candidate_path: Path,
    flowtwin_path: Path,
    tcn_path: Path,
    dspr_path: Path,
    tcn_alarm_audit_path: Path,
    output_path: Path,
) -> Path:
    sources = {
        VARIANT_ORDER[0]: _load_source(candidate_path, "candidate"),
        VARIANT_ORDER[1]: _load_source(flowtwin_path, "flowtwin"),
        VARIANT_ORDER[3]: _load_source(dspr_path, "dspr"),
    }
    candidate = sources[VARIANT_ORDER[0]]
    candidate_contract = candidate["candidate_contract"]
    if (
        candidate_contract.get("candidate_id") != VARIANT_ORDER[0]
        or tuple(candidate_contract.get("planned_comparison_variants", ()))
        != VARIANT_ORDER
    ):
        raise ValueError("candidate contract does not declare the exact four-model order")
    expected_budget = candidate_contract["training_budget"]
    fingerprints: list[dict[str, Any]] = []
    records_by_variant: dict[str, dict[str, Any]] = {}
    for variant, source in sources.items():
        _validate_development_state(source)
        if source["candidate_contract"] != candidate_contract:
            raise ValueError("candidate-contract operational sources differ")
        if source["config"].get("budget") != expected_budget:
            raise ValueError(f"{variant} budget differs from candidate contract")
        fingerprints.append(_binding_fingerprint(source))
        records_by_variant[variant] = _only_run(source, variant)
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise ValueError("operational source input bindings differ")
    _validate_execution_compatibility(candidate, sources[VARIANT_ORDER[1]])
    _validate_execution_compatibility(candidate, sources[VARIANT_ORDER[3]])

    tcn = _load_raw_tcn(tcn_path)
    if tcn["contract"] != candidate["contract"]:
        raise ValueError("TCN and candidate benchmark contracts differ")
    if tcn["config"].get("budget") != expected_budget:
        raise ValueError("TCN raw budget differs from candidate contract")
    if _input_fingerprint(tcn["manifest"]) != _input_fingerprint(candidate["manifest"]):
        raise ValueError("TCN raw input binding differs")
    if tcn["config"].get("benchmark_contract_sha256") != fingerprints[0][
        "benchmark_contract_sha256"
    ]:
        raise ValueError("TCN base benchmark SHA differs")
    _common_execution_without_candidate(candidate, tcn)
    tcn_record = _only_run(tcn, "tcn")
    records_by_variant["tcn"] = tcn_record

    records = [records_by_variant[name] for name in VARIANT_ORDER]
    seeds = {int(record["seed"]) for record in records}
    if seeds != set(expected_budget["seeds"]):
        raise ValueError("four-model seed set differs from candidate contract")
    profile_signature = _diagnostic_profile_signature(records)
    # Operational sources must also have complete operational endpoints.
    for variant in (VARIANT_ORDER[0], VARIANT_ORDER[1], VARIANT_ORDER[3]):
        _profile_signature(records_by_variant[variant], variant)

    feasibility = [
        _selected_feasibility(sources[VARIANT_ORDER[0]], records_by_variant[VARIANT_ORDER[0]]),
        _selected_feasibility(sources[VARIANT_ORDER[1]], records_by_variant[VARIANT_ORDER[1]]),
    ]
    tcn_feasibility, tcn_audit_source = _validate_tcn_audit(
        tcn_alarm_audit_path, tcn, tcn_record
    )
    feasibility.append(tcn_feasibility)
    feasibility.append(
        _selected_feasibility(sources[VARIANT_ORDER[3]], records_by_variant[VARIANT_ORDER[3]])
    )
    diagnostics = [_diagnostic_row(record) for record in records]
    conditional_records = [
        records_by_variant[VARIANT_ORDER[0]],
        records_by_variant[VARIANT_ORDER[1]],
        records_by_variant[VARIANT_ORDER[3]],
    ]
    conditional = [_conditional_operational_row(record) for record in conditional_records]

    diagnostic_pairs = _paired_comparisons(
        records, DIAGNOSTIC_ENDPOINTS, VARIANT_ORDER[1:], candidate["contract"]
    )
    operational_pairs = _paired_comparisons(
        conditional_records,
        OPERATIONAL_ENDPOINTS,
        (VARIANT_ORDER[1], VARIANT_ORDER[3]),
        candidate["contract"],
    )
    source_artifacts = {
        variant: {
            "path": str(source["root"]),
            "checksums_sha256": sha256_file(Path(source["root"]) / "checksums.sha256"),
            "benchmark_results_sha256": sha256_file(
                Path(source["root"]) / "benchmark_results.json"
            ),
        }
        for variant, source in {**sources, "tcn": tcn}.items()
    }
    source_artifacts["tcn_alarm_validation_audit"] = tcn_audit_source
    report = {
        "report_version": REPORT_VERSION,
        "status": "post_hoc_development_only",
        "d2_test_seen_before_freeze": True,
        "eligible_for_confirmatory_claims": False,
        "variant_order": list(VARIANT_ORDER),
        "budget": expected_budget,
        "diagnostic_table": diagnostics,
        "diagnostic_table_role": "descriptive_pooled_not_inferential",
        "operational_feasibility": feasibility,
        "conditional_operational_table": conditional,
        "conditional_operational_table_role": (
            "test results only for variants passing the frozen validation gate; "
            "TCN remains N/A and is not ranked as zero"
        ),
        "candidate_minus_comparator_profile_paired_diagnostic": diagnostic_pairs,
        "candidate_minus_comparator_profile_paired_operational": operational_pairs,
        "complete_profile_signature": profile_signature,
        "timing_comparability": {
            "parameter_count": True,
            "training_wall_time_and_throughput": False,
            "reason": "model runs overlapped as independent CPU processes",
        },
        "metric_naming": {
            "validation_fitted_row": (
                "probability calibration and a validation-selected point threshold; "
                "not an uncalibrated raw model output"
            ),
            "operational": "validation-selected hysteretic state-machine alarm",
        },
        "source_artifacts": source_artifacts,
        "claim_boundary": candidate_contract["claim_boundary"],
        "future_confirmatory_gate": candidate_contract["future_confirmatory_gate"],
    }

    temporary, existed_empty = _atomic_output(output_path)
    try:
        write_json(temporary / "development_report.json", report)
        _write_csv(temporary / "diagnostic_table.csv", diagnostics)
        _write_csv(temporary / "operational_feasibility.csv", feasibility)
        _write_csv(temporary / "conditional_operational_table.csv", conditional)
        manifest = {
            "report_version": REPORT_VERSION,
            "status": report["status"],
            "eligible_for_confirmatory_claims": False,
            "variant_order": list(VARIANT_ORDER),
            "candidate_contract_sha256": fingerprints[0]["candidate_contract_sha256"],
            "benchmark_contract_sha256": fingerprints[0]["benchmark_contract_sha256"],
            "source_artifacts": source_artifacts,
            "reporter_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "validation_helper_source": {
                "path": str(VALIDATION_HELPER_SOURCE),
                "sha256": sha256_file(VALIDATION_HELPER_SOURCE),
            },
            "aggregation_runner_source": {
                "path": str((Path(__file__).resolve().parent / "run_flowtwin_benchmark.py")),
                "sha256": sha256_file(
                    Path(__file__).resolve().parent / "run_flowtwin_benchmark.py"
                ),
            },
            "timing_comparability": report["timing_comparability"],
            "artifacts": [*OUTPUT_FILES, "checksums.sha256"],
            "claim_boundary": report["claim_boundary"],
        }
        write_json(temporary / "run_manifest.json", manifest)
        write_checksums(temporary, OUTPUT_FILES)
        output = Path(output_path).resolve()
        if existed_empty:
            output.rmdir()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Path(output_path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--flowtwin", type=Path, required=True)
    parser.add_argument("--tcn", type=Path, required=True)
    parser.add_argument("--dspr", type=Path, required=True)
    parser.add_argument("--tcn-alarm-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = build_report(
            candidate_path=args.candidate,
            flowtwin_path=args.flowtwin,
            tcn_path=args.tcn,
            dspr_path=args.dspr,
            tcn_alarm_audit_path=args.tcn_alarm_audit,
            output_path=args.output,
        )
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(f"FlowTwin v0.3 development report failed: {exc}") from exc
    print(f"FlowTwin v0.3 development report complete: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
