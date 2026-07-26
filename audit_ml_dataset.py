#!/usr/bin/env python3
"""Audit an HTST ML dataset for schema, provenance and temporal leakage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ml_pipeline_common import (
    BASE_DIR,
    CONTRACT_PATH,
    feature_fields,
    load_contract,
    read_csv_header,
    read_csv_rows,
    sha256_file,
    taxonomy_indexes,
    verify_checksums,
    write_json,
)


AUDITOR_VERSION = "2.2.0"
EXPLICIT_FORBIDDEN = {
    "alarm_count",
    "observable_alarm_count",
    "alarm_unsafe_forward",
    "diagnostic_alarm_count",
    "oracle_alarm_count",
    "total_alarm_count",
    "differential_pressure_bar",
    "fdv_actual_forward",
    "sensor_available",
    "valve_travel_time_factor",
    "valve_leakage_fraction",
    "cleaning_effectiveness_factor",
    "balance_tank_product_fraction",
    "holding_chemical_fraction",
    "holding_product_fraction",
    "routed_chemical_fraction",
    "routed_product_fraction",
    "routed_hygiene_risk_fraction",
    "forward_chemical_fraction",
    "forward_product_fraction",
    "forward_hygiene_risk_fraction",
    "surface_total_soil_g",
    "surface_hygiene_risk_fraction",
}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if not condition:
            self.errors.append(f"{name}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _csv_value_matches_type(value: str, declared: str, nullable: bool) -> bool:
    if value == "":
        return nullable
    try:
        if declared == "integer":
            int(value)
        elif declared == "number":
            return math.isfinite(float(value))
        elif declared == "boolean":
            return value.lower() in {"true", "false", "0", "1"}
        elif declared != "string":
            return False
    except ValueError:
        return False
    return True


def _validate_declared_table_schema(
    path: Path, table_schema: Mapping[str, Any]
) -> tuple[bool, str]:
    declared_fields = table_schema.get("fields")
    if not isinstance(declared_fields, list) or not declared_fields:
        return False, f"{path.name}: fields contract missing"
    declared_names = [str(field.get("name", "")) for field in declared_fields]
    actual_names = read_csv_header(path)
    if declared_names != actual_names:
        return False, f"{path.name}: declared/actual headers differ"
    declarations = {str(field["name"]): field for field in declared_fields}
    for row_index, row in enumerate(read_csv_rows(path), start=2):
        for name in actual_names:
            declaration = declarations[name]
            if not _csv_value_matches_type(
                row[name],
                str(declaration.get("type", "")),
                bool(declaration.get("nullable", False)),
            ):
                return (
                    False,
                    f"{path.name}:{row_index}:{name} violates "
                    f"type={declaration.get('type')} "
                    f"nullable={declaration.get('nullable')}",
                )
    return True, f"{path.name}: header, type and nullable contract valid"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def _one_split_per_value(
    rows: Iterable[Mapping[str, str]], value_field: str
) -> tuple[bool, list[str]]:
    owners: dict[str, str] = {}
    conflicts: set[str] = set()
    for row in rows:
        value = row[value_field]
        split = row["split"]
        previous = owners.setdefault(value, split)
        if previous != split:
            conflicts.add(value)
    return not conflicts, sorted(conflicts)


def _audit_windows(
    audit: Audit,
    path: Path | None,
    episode_split: Mapping[str, str],
) -> dict[str, Any]:
    if path is None:
        audit.warn("No window manifest supplied; window-overlap audit is not applicable yet")
        return {"status": "not_applicable", "window_count": 0}
    rows = list(read_csv_rows(path))
    required = {"window_id", "episode_id", "window_start_s", "window_end_s"}
    header = set(read_csv_header(path))
    audit.check("window_schema", required <= header, f"required={sorted(required)}")
    if not required <= header:
        return {"status": "failed", "window_count": len(rows)}
    seen_ids: set[str] = set()
    seen_ranges: set[tuple[str, float, float]] = set()
    valid = True
    by_episode: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        try:
            start = float(row["window_start_s"])
            end = float(row["window_end_s"])
        except ValueError:
            valid = False
            continue
        episode = row["episode_id"]
        valid = valid and row["window_id"] not in seen_ids and start < end
        seen_ids.add(row["window_id"])
        key = (episode, start, end)
        valid = valid and key not in seen_ranges and episode in episode_split
        seen_ranges.add(key)
        assigned = row.get("split", episode_split.get(episode, ""))
        valid = valid and assigned == episode_split.get(episode)
        by_episode[episode].append((start, end, assigned))
    overlap_cross_split = False
    for windows in by_episode.values():
        windows.sort()
        for index, (start, end, split) in enumerate(windows):
            for other_start, other_end, other_split in windows[index + 1 :]:
                if other_start >= end:
                    break
                if start < other_end and other_start < end and split != other_split:
                    overlap_cross_split = True
    valid = valid and not overlap_cross_split
    audit.check(
        "window_overlap_isolation",
        valid,
        "unique valid windows; overlapping windows never cross a split",
    )
    return {"status": "passed" if valid else "failed", "window_count": len(rows)}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = Audit()
    contract = load_contract()
    dataset_checksum_failures = verify_checksums(args.dataset)
    split_checksum_failures = verify_checksums(args.splits)
    audit.check(
        "dataset_checksums",
        not dataset_checksum_failures,
        "; ".join(dataset_checksum_failures) or "all dataset checksums match",
    )
    audit.check(
        "split_checksums",
        not split_checksum_failures,
        "; ".join(split_checksum_failures) or "all split checksums match",
    )

    manifest_path = args.dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest_path = args.splits / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (args.dataset / "dataset_schema.json").read_text(encoding="utf-8")
    )
    schema_problems: list[str] = []
    for table_name in (
        "profiles.csv",
        "episodes.csv",
        "signals.csv",
        "oracle_labels.csv",
    ):
        valid, detail = _validate_declared_table_schema(
            args.dataset / table_name,
            schema.get("tables", {}).get(table_name, {}),
        )
        if not valid:
            schema_problems.append(detail)
    audit.check(
        "declared_table_schema",
        not schema_problems,
        "; ".join(schema_problems) or "all four table schemas match their CSV files",
    )
    audit.check(
        "v2_version_contract",
        manifest.get("contract_version") == contract["contract_version"] == "2.2.0"
        and manifest.get("model_version") == contract["model_version"] == "2.2.0"
        and str(manifest.get("generator_version", "")).startswith("2.")
        and schema.get("schema_version") == "2.2.0"
        and str(split_manifest.get("splitter_version", "")).startswith("2."),
        "dataset, schema, contract, model, generator and splitter must all use v2",
    )
    audit.check(
        "split_source_manifest",
        split_manifest.get("dataset_manifest_sha256") == sha256_file(manifest_path),
        "split manifest is bound to this exact dataset manifest",
    )

    signal_header = read_csv_header(args.dataset / "signals.csv")
    label_header = read_csv_header(args.dataset / "oracle_labels.csv")
    context = set(contract["signal_context_fields"])
    candidates = set(signal_header) - context
    allowed_union = {
        field for fields in contract["feature_sets"].values() for field in fields
    }
    audit.check(
        "explicit_privileged_fields_absent",
        not (EXPLICIT_FORBIDDEN & set(signal_header)),
        f"forbidden columns found={sorted(EXPLICIT_FORBIDDEN & set(signal_header))}",
    )
    forbidden_candidates = set(contract["always_forbidden_features"]) & candidates
    audit.check(
        "feature_forbidden_allowlist",
        not forbidden_candidates and candidates <= allowed_union,
        f"forbidden={sorted(forbidden_candidates)}, unknown={sorted(candidates - allowed_union)}",
    )
    expected_features = set(manifest.get("model_feature_fields", []))
    audit.check(
        "manifest_feature_contract",
        candidates == expected_features == set(feature_fields(contract, "S3-context")),
        "signals, manifest, and S3-context feature fields agree exactly",
    )
    alarm_fields = set(contract.get("diagnostic_alarm_fields", []))
    audit.check(
        "diagnostic_alarms_separated",
        not (alarm_fields & set(signal_header)),
        f"alarm columns in signals={sorted(alarm_fields & set(signal_header))}",
    )
    allowed_overlap = {"episode_id", "time_start_s", "time_s"}
    unexpected_overlap = (set(signal_header) & set(label_header)) - allowed_overlap
    audit.check(
        "signal_label_separation",
        not unexpected_overlap,
        f"unexpected table overlap={sorted(unexpected_overlap)}",
    )
    audit.check(
        "actual_dp_is_oracle_only",
        "differential_pressure_bar" in label_header
        and "measured_differential_pressure_bar" in signal_header,
        "actual dP must be a label and measured dP must be the feature",
    )
    v2_truth_fields = {
        "mean_transport_residence_time_s",
        "fdv_position",
        "power_available",
        "sensor_available",
        "valve_travel_time_factor",
        "valve_leakage_fraction",
        "cleaning_effectiveness_factor",
        "fdv_actual_forward",
        "balance_tank_volume_l",
        "balance_tank_temp_c",
        "balance_tank_mean_pass_count",
        "balance_tank_risk_fraction",
        "balance_tank_chemical_fraction",
        "balance_tank_product_fraction",
        "holding_chemical_fraction",
        "holding_product_fraction",
        "routed_chemical_fraction",
        "routed_product_fraction",
        "routed_hygiene_risk_fraction",
        "forward_chemical_fraction",
        "forward_product_fraction",
        "forward_hygiene_risk_fraction",
        "surface_protein_soil_g",
        "surface_mineral_soil_g",
        "surface_total_soil_g",
        "surface_hygiene_risk_fraction",
        "cip_protein_soil_g",
        "cip_mineral_soil_g",
        "cip_total_soil_g",
        "cip_residual_chemical_fraction",
        "cip_cleaning_complete",
        "shadow_regenerator_wall_temp_c",
        "shadow_heater_wall_temp_c",
        "shadow_cooler_wall_temp_c",
        "physical_effect_time_s",
    }
    audit.check(
        "v2_truth_fields_are_oracle_only",
        v2_truth_fields <= set(label_header)
        and not (v2_truth_fields & set(signal_header)),
        f"missing labels={sorted(v2_truth_fields - set(label_header))}, "
        f"leaked signals={sorted(v2_truth_fields & set(signal_header))}",
    )

    episodes = list(read_csv_rows(args.dataset / "episodes.csv"))
    profiles = list(read_csv_rows(args.dataset / "profiles.csv"))
    splits = list(read_csv_rows(args.splits / "episode_splits.csv"))
    episode_ids = [row["episode_id"] for row in episodes]
    profile_ids = [row["plant_profile_id"] for row in profiles]
    audit.check(
        "episode_primary_key",
        bool(episode_ids) and len(episode_ids) == len(set(episode_ids)),
        f"rows={len(episode_ids)}, unique={len(set(episode_ids))}",
    )
    audit.check(
        "profile_primary_key",
        bool(profile_ids) and len(profile_ids) == len(set(profile_ids)),
        f"rows={len(profile_ids)}, unique={len(set(profile_ids))}",
    )
    split_episode_ids = [row["episode_id"] for row in splits]
    audit.check(
        "split_episode_coverage",
        len(split_episode_ids) == len(set(split_episode_ids))
        and set(split_episode_ids) == set(episode_ids),
        "every episode appears exactly once in the split manifest",
    )
    for field in (
        "plant_profile_id",
        "counterfactual_group_id",
        "profile_config_hash",
        "noise_seed",
        "sampling_seed",
    ):
        clean, conflicts = _one_split_per_value(splits, field)
        audit.check(
            f"split_isolation_{field}",
            clean,
            f"cross-split values={conflicts[:10]}",
        )

    by_code, _ = taxonomy_indexes(contract)
    taxonomy_valid = True
    taxonomy_problems: list[str] = []
    for row in episodes:
        item = by_code.get(row["canonical_code"])
        if (
            item is None
            or not item["implemented"]
            or row["canonical_scenario"] != item["canonical_scenario"]
            or row["simulator_scenario"] != item["canonical_scenario"]
            or row["simulator_scenario"] not in item.get("simulator_scenarios", [])
            or row["simulator_scenario"] in item.get("aliases", [])
            or row["model_version"] != manifest.get("model_version")
            or row["contract_version"] != manifest.get("contract_version")
            or row["generator_version"] != manifest.get("generator_version")
        ):
            taxonomy_valid = False
            taxonomy_problems.append(row["episode_id"])
    audit.check(
        "canonical_taxonomy",
        taxonomy_valid,
        f"invalid/unimplemented/legacy-alias episodes={taxonomy_problems[:10]}",
    )
    effect_order_problems: list[str] = []
    for row in episodes:
        injection_text = row.get("behavior_injection_time_s", "")
        physical_text = row.get("physical_effect_time_s", "")
        observable_text = row.get("behavior_effect_time_s", "")
        if not (injection_text or physical_text or observable_text):
            continue
        if not (injection_text and physical_text):
            effect_order_problems.append(row["episode_id"])
            continue
        injection = float(injection_text)
        physical = float(physical_text)
        if injection > physical + 1e-12:
            effect_order_problems.append(row["episode_id"])
            continue
        if observable_text:
            observable = float(observable_text)
            if physical > observable + 1e-12:
                effect_order_problems.append(row["episode_id"])
        elif row.get("detection_eligible") == "1":
            effect_order_problems.append(row["episode_id"])
    audit.check(
        "physical_before_observable_effect",
        not effect_order_problems,
        "injection <= physical effect <= observable effect; invalid episodes="
        f"{effect_order_problems[:10]}",
    )
    absent_codes = {
        item["code"] for item in contract["taxonomy"] if not item["implemented"]
    }
    observed_codes = {row["canonical_code"] for row in episodes}
    audit.check(
        "unimplemented_classes_absent",
        not (absent_codes & observed_codes),
        f"unimplemented codes found={sorted(absent_codes & observed_codes)}",
    )
    selected_codes = {
        item["code"] for item in manifest.get("selected_scenarios", [])
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        groups[row["counterfactual_group_id"]].append(row)
    counterfactual_valid = True
    counterfactual_problems: list[str] = []
    for group_id, rows in groups.items():
        constant_fields = (
            "plant_profile_id",
            "profile_config_hash",
            "noise_seed",
            "sampling_seed",
            "duration_s",
            "dt_s",
        )
        codes = [row["canonical_code"] for row in rows]
        valid = (
            all(len({row[field] for row in rows}) == 1 for field in constant_fields)
            and len(codes) == len(set(codes))
            and set(codes) == selected_codes
        )
        fault_rows_in_group = [row for row in rows if row["is_fault"] == "1"]
        if fault_rows_in_group:
            normal_reference_faults = [
                row
                for row in fault_rows_in_group
                if row.get("counterfactual_reference", "normal") == "normal"
            ]
            for field in (
                "fault_injection_time_s",
                "fault_end_time_s",
                "fault_duration_s",
                "fault_severity",
            ):
                valid = valid and (
                    not normal_reference_faults
                    or len({row[field] for row in normal_reference_faults}) == 1
                )
            for row in fault_rows_in_group:
                try:
                    injection = float(row["fault_injection_time_s"])
                    end = float(row["fault_end_time_s"])
                    physical_effect = float(row["physical_effect_time_s"])
                    duration = float(row["duration_s"])
                    eligible = row["detection_eligible"] == "1"
                    if eligible:
                        effect = float(row["fault_effect_time_s"])
                        valid = valid and injection <= effect <= duration
                    else:
                        valid = valid and not row["fault_effect_time_s"]
                    valid = valid and 0.0 <= injection <= physical_effect <= duration
                    valid = valid and injection < end <= duration
                    valid = valid and row["behavior_injection_time_s"] == row[
                        "fault_injection_time_s"
                    ]
                    valid = valid and row["behavior_end_time_s"] == row[
                        "fault_end_time_s"
                    ]
                    valid = valid and row["counterfactual_reference"] in {
                        "normal",
                        "cip_cycle",
                    }
                    valid = valid and bool(
                        row["counterfactual_reference_config_hash"]
                    )
                except ValueError:
                    valid = False
        for row in rows:
            has_behavior = row["canonical_code"] != "N00"
            if has_behavior:
                try:
                    injection = float(row["behavior_injection_time_s"])
                    physical_effect = float(row["physical_effect_time_s"])
                    duration = float(row["duration_s"])
                    valid = valid and 0.0 <= injection <= physical_effect <= duration
                    if row["detection_eligible"] == "1":
                        effect = float(row["behavior_effect_time_s"])
                        valid = valid and injection <= effect <= duration
                except ValueError:
                    valid = False
            else:
                valid = valid and all(
                    not row[field]
                    for field in (
                        "behavior_injection_time_s",
                        "behavior_end_time_s",
                        "behavior_effect_time_s",
                        "physical_effect_time_s",
                    )
                )
        if not valid:
            counterfactual_valid = False
            counterfactual_problems.append(group_id)
    audit.check(
        "counterfactual_group_integrity",
        counterfactual_valid,
        f"invalid groups={counterfactual_problems[:10]}",
    )

    signal_path = args.dataset / "signals.csv"
    label_path = args.dataset / "oracle_labels.csv"
    aligned = True
    temporal_labels_valid = True
    temporal_label_problems: set[str] = set()
    future_censoring_valid = True
    future_censoring_problems: set[str] = set()
    row_count = 0
    per_episode_counts: Counter[str] = Counter()
    last_time: dict[str, float] = {}
    previous_mode: dict[str, str] = {}
    episode_by_id = {row["episode_id"]: row for row in episodes}
    future_horizons = {
        field: float(field.rsplit("_", 1)[1][:-1])
        for field in label_header
        if field.startswith("future_") and field.endswith("s")
    }
    with (
        signal_path.open(newline="", encoding="utf-8") as signal_handle,
        label_path.open(newline="", encoding="utf-8") as label_handle,
    ):
        signal_reader = csv.DictReader(signal_handle)
        label_reader = csv.DictReader(label_handle)
        while True:
            signal = next(signal_reader, None)
            label = next(label_reader, None)
            if signal is None or label is None:
                aligned = aligned and signal is None and label is None
                break
            row_count += 1
            keys = ("episode_id", "time_start_s", "time_s")
            if any(signal[key] != label[key] for key in keys):
                aligned = False
            episode = signal["episode_id"]
            meta = episode_by_id[episode]
            time_s = float(signal["time_s"])
            time_start_s = float(signal["time_start_s"])
            if episode in last_time and time_s <= last_time[episode]:
                aligned = False
            last_time[episode] = time_s
            per_episode_counts[episode] += 1
            row_valid = label["detection_eligible"] == meta["detection_eligible"]
            fault_active = label["fault_active"] == "1"
            behavior_active = label["behavior_active"] == "1"
            is_fault = meta["is_fault"] == "1"
            code = meta["canonical_code"]
            row_valid = row_valid and (is_fault or not fault_active)
            if fault_active:
                injection = float(meta["fault_injection_time_s"])
                end = float(meta["fault_end_time_s"])
                row_valid = row_valid and time_start_s + 1e-12 >= injection
                row_valid = row_valid and time_start_s < end - 1e-12
            if code == "N00":
                row_valid = row_valid and not behavior_active
            elif code == "M02":
                row_valid = row_valid and behavior_active
            elif behavior_active:
                injection = float(meta["behavior_injection_time_s"])
                end = float(meta["behavior_end_time_s"])
                row_valid = row_valid and time_start_s + 1e-12 >= injection
                row_valid = row_valid and time_start_s < end - 1e-12
            if is_fault:
                row_valid = row_valid and behavior_active == fault_active
            expected_mode_transition = (
                by_code[code]["kind"] == "mode"
                and episode in previous_mode
                and signal["plant_mode"] != previous_mode[episode]
            )
            row_valid = row_valid and (
                (label["mode_transition_event"] == "1")
                == expected_mode_transition
            )
            previous_mode[episode] = signal["plant_mode"]
            duration_s = float(meta["duration_s"])
            for field, horizon_s in future_horizons.items():
                censored = time_s + horizon_s > duration_s + 1e-12
                value = label[field]
                valid_value = (not value) if censored else value in {"0", "1"}
                if not valid_value:
                    future_censoring_valid = False
                    future_censoring_problems.add(episode)
            if not row_valid:
                temporal_labels_valid = False
                temporal_label_problems.add(episode)
    audit.check(
        "row_key_alignment",
        aligned,
        f"signals and labels align monotonically for {row_count} rows",
    )
    audit.check(
        "counterfactual_effect_and_activity_timing",
        temporal_labels_valid,
        f"invalid episodes={sorted(temporal_label_problems)[:10]}",
    )
    audit.check(
        "future_label_right_censoring",
        future_censoring_valid,
        f"invalid episodes={sorted(future_censoring_problems)[:10]}",
    )
    expected_counts = {row["episode_id"]: int(row["row_count"]) for row in episodes}
    audit.check(
        "episode_row_counts",
        dict(per_episode_counts) == expected_counts,
        "per-episode row counts match episodes.csv",
    )
    manifest_counts = manifest.get("counts", {})
    audit.check(
        "manifest_counts",
        manifest_counts.get("episodes") == len(episodes)
        and manifest_counts.get("signal_rows") == row_count
        and manifest_counts.get("label_rows") == row_count,
        f"manifest={manifest_counts}, observed episodes={len(episodes)}, rows={row_count}",
    )

    fault_rows = [row for row in episodes if row["is_fault"] == "1"]
    diversity = {
        "noise_seed": len({row["noise_seed"] for row in episodes}),
        "fault_injection_time_s": len(
            {row["fault_injection_time_s"] for row in fault_rows}
        ),
        "fault_severity": len({row["fault_severity"] for row in fault_rows}),
    }
    if len({row["counterfactual_group_id"] for row in episodes}) > 1:
        audit.check(
            "episode_randomization",
            all(value > 1 for value in diversity.values()),
            f"independent-value counts={diversity}",
        )
    else:
        audit.warn("One counterfactual group cannot demonstrate randomized diversity")

    episode_split = {row["episode_id"]: row["split"] for row in splits}
    window_audit = _audit_windows(audit, args.windows, episode_split)
    split_counts = Counter(row["split"] for row in splits)
    report = {
        "auditor_version": AUDITOR_VERSION,
        "status": "passed" if not audit.errors else "failed",
        "dataset_id": manifest.get("dataset_version"),
        "splitter_version": split_manifest.get("splitter_version"),
        "checks": audit.checks,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "provenance": {
            "audit_ml_dataset.py": sha256_file(Path(__file__).resolve()),
            "ml_pipeline_common.py": sha256_file(BASE_DIR / "ml_pipeline_common.py"),
            "ml_contract.json": sha256_file(CONTRACT_PATH),
            "dataset_manifest.json": sha256_file(manifest_path),
            "split_manifest.json": sha256_file(split_manifest_path),
        },
        "summary": {
            "profiles": len(profiles),
            "episodes": len(episodes),
            "rows": row_count,
            "split_episode_counts": dict(sorted(split_counts.items())),
            "randomization_unique_counts": diversity,
            "window_audit": window_audit,
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_audit(args)
    report_path = args.report or args.dataset / "audit_report.json"
    write_json(report_path, report)
    checksum_path = report_path.with_name(report_path.name + ".sha256")
    checksum_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        f"Audit {report['status']}: {len(report['checks'])} checks, "
        f"{len(report['errors'])} failures; report={report_path}; "
        f"checksum={checksum_path}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
