#!/usr/bin/env python3
"""Audit D3-RUL provenance, censor semantics, EOL evidence and split isolation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        EPSILON,
        evaluate_eol_trace,
        feature_fields,
        load_d3_contract,
        model_version_is_compatible,
        read_csv_header,
        read_csv_rows,
        sha256_file,
        sha256_json,
        stable_seed,
        validate_declared_schema,
        verify_checksums,
        write_json,
    )
except ImportError:  # pragma: no cover
    from d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        EPSILON,
        evaluate_eol_trace,
        feature_fields,
        load_d3_contract,
        model_version_is_compatible,
        read_csv_header,
        read_csv_rows,
        sha256_file,
        sha256_json,
        stable_seed,
        validate_declared_schema,
        verify_checksums,
        write_json,
    )


AUDITOR_VERSION = "1.2.0"


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "pass" if condition else "fail", "detail": detail}
        )
        if not condition:
            self.errors.append(f"{name}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def _one_split(rows: Iterable[Mapping[str, str]], field: str) -> list[str]:
    owners: dict[str, str] = {}
    conflicts: set[str] = set()
    for row in rows:
        value = row[field]
        previous = owners.setdefault(value, row["split"])
        if previous != row["split"]:
            conflicts.add(value)
    return sorted(conflicts)


def _rotated_choice(values: Sequence[float], seed: int, position: int) -> float:
    offset = random.Random(seed).randrange(len(values))
    return float(values[(offset + position) % len(values)])


def _expected_profile_values(
    contract: Mapping[str, Any], seed: int, domain: str
) -> dict[str, float]:
    ranges = dict(contract["profile_parameter_ranges"])
    if domain == "OOD":
        ranges.update(contract["ood_profile_parameter_ranges"])
    rng = random.Random(seed)
    return {
        name: rng.uniform(float(bounds[0]), float(bounds[1]))
        for name, bounds in sorted(ranges.items())
    }


def _close(left: float | str, right: float | str, tolerance: float = 1.0e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=tolerance)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = Audit()
    contract = load_d3_contract()
    dataset_failures = verify_checksums(args.dataset)
    split_failures = verify_checksums(args.splits)
    audit.check(
        "dataset_checksums",
        not dataset_failures,
        "; ".join(dataset_failures) or "all dataset checksums match",
    )
    audit.check(
        "split_checksums",
        not split_failures,
        "; ".join(split_failures) or "all split checksums match",
    )
    manifest_path = args.dataset / "dataset_manifest.json"
    split_manifest_path = args.splits / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    schema = json.loads((args.dataset / "dataset_schema.json").read_text(encoding="utf-8"))
    audit.check(
        "d3_version_contract",
        manifest.get("dataset_family") == schema.get("dataset_family") == "D3-RUL"
        and manifest.get("contract_version") == contract["contract_version"]
        and schema.get("schema_version") == contract["contract_version"]
        and model_version_is_compatible(str(manifest.get("model_version", "")), contract)
        and str(manifest.get("generator_version", "")).startswith("1.")
        and str(split_manifest.get("splitter_version", "")).startswith("1."),
        "D3 contract/schema/scripts must be compatible and runtime model major must be allowed",
    )
    parameter_metadata = manifest.get("parameters", {})
    audit.check(
        "eol_evaluation_grid_contract",
        contract.get("eol", {}).get("evaluation_grid")
        == "emitted_sample_interval_s"
        and "sample_interval_s" in parameter_metadata
        and "eol_evaluation_resolution_sim_s" in parameter_metadata
        and _close(
            parameter_metadata["sample_interval_s"],
            parameter_metadata["eol_evaluation_resolution_sim_s"],
        ),
        "EOL is evaluated at the declared emitted-row resolution so it can be independently reproduced",
    )
    audit.check(
        "split_source_manifest",
        split_manifest.get("dataset_manifest_sha256") == sha256_file(manifest_path),
        "split manifest is bound to the exact dataset manifest",
    )
    provenance_valid = (
        manifest.get("provenance", {}).get("d3_rul_contract.json")
        == sha256_file(CONTRACT_PATH)
        and manifest.get("provenance", {}).get("d3_rul_common.py")
        == sha256_file(BASE_DIR / "d3_rul_common.py")
    )
    audit.check(
        "contract_and_common_provenance",
        provenance_valid,
        "manifest hashes bind the D3 contract and common label implementation",
    )

    table_names = (
        "profiles.csv",
        "trajectories.csv",
        "signals.csv",
        "oracle_labels.csv",
        "eol_evidence.csv",
    )
    schema_problems: list[str] = []
    for name in table_names:
        valid, detail = validate_declared_schema(
            args.dataset / name, schema.get("tables", {}).get(name, {}).get("fields", [])
        )
        if not valid:
            schema_problems.append(detail)
    audit.check(
        "declared_table_schema",
        not schema_problems,
        "; ".join(schema_problems) or "all five D3 table schemas match",
    )
    signal_declarations = {
        str(item.get("name")): str(item.get("role"))
        for item in schema.get("tables", {})
        .get("signals.csv", {})
        .get("fields", [])
    }
    expected_context = set(contract["signal_context_fields"])
    expected_features = set(manifest.get("model_feature_fields", []))
    signal_roles_valid = (
        expected_context.isdisjoint(expected_features)
        and all(signal_declarations.get(name) == "context" for name in expected_context)
        and all(
            signal_declarations.get(name) == "observable_feature"
            for name in expected_features
        )
        and set(signal_declarations) == expected_context | expected_features
    )
    audit.check(
        "signal_schema_roles",
        signal_roles_valid,
        "signal context fields and model features must have disjoint declared roles",
    )

    signal_header = read_csv_header(args.dataset / "signals.csv")
    label_header = read_csv_header(args.dataset / "oracle_labels.csv")
    context = set(contract["signal_context_fields"])
    signal_candidates = set(signal_header) - context
    expected_features = set(feature_fields(contract))
    forbidden = set(contract["always_forbidden_signal_fields"])
    audit.check(
        "signal_feature_allowlist",
        signal_candidates == expected_features
        and not (forbidden & set(signal_header))
        and set(manifest.get("model_feature_fields", [])) == expected_features,
        f"unknown={sorted(signal_candidates - expected_features)}, "
        f"missing={sorted(expected_features - signal_candidates)}, "
        f"forbidden={sorted(forbidden & set(signal_header))}",
    )
    allowed_overlap = {"trajectory_id", "time_start_sim_s", "time_sim_s"}
    unexpected_overlap = (set(signal_header) & set(label_header)) - allowed_overlap
    audit.check(
        "signal_oracle_separation",
        not unexpected_overlap,
        f"unexpected shared fields={sorted(unexpected_overlap)}",
    )

    profiles = list(read_csv_rows(args.dataset / "profiles.csv"))
    trajectories = list(read_csv_rows(args.dataset / "trajectories.csv"))
    splits = list(read_csv_rows(args.splits / "trajectory_splits.csv"))
    evidence_rows = list(read_csv_rows(args.dataset / "eol_evidence.csv"))
    profile_ids = [row["plant_profile_id"] for row in profiles]
    trajectory_ids = [row["trajectory_id"] for row in trajectories]
    audit.check(
        "primary_keys",
        bool(profile_ids)
        and len(profile_ids) == len(set(profile_ids))
        and bool(trajectory_ids)
        and len(trajectory_ids) == len(set(trajectory_ids)),
        f"profiles={len(profile_ids)}/{len(set(profile_ids))}, "
        f"trajectories={len(trajectory_ids)}/{len(set(trajectory_ids))}",
    )
    split_ids = [row["trajectory_id"] for row in splits]
    audit.check(
        "split_trajectory_coverage",
        len(split_ids) == len(set(split_ids)) and set(split_ids) == set(trajectory_ids),
        "every trajectory must appear in exactly one split row",
    )
    for field in (
        "plant_profile_id",
        "life_family_id",
        "profile_config_hash",
        "noise_seed",
        "degradation_seed",
        "censor_seed",
    ):
        conflicts = _one_split(splits, field)
        audit.check(
            f"split_isolation_{field}",
            not conflicts,
            f"cross-split values={conflicts[:10]}",
        )

    profile_by_id = {row["plant_profile_id"]: row for row in profiles}
    trajectory_by_id = {row["trajectory_id"]: row for row in trajectories}
    parameters = manifest["parameters"]
    master_seed = int(manifest["seed"])
    accelerations = [float(value) for value in parameters["accelerations"]]
    horizons = [float(value) for value in parameters["censor_horizons_equivalent_h"]]
    profile_contract_valid = True
    trajectory_sampling_valid = True
    outcome_metadata_valid = True
    profile_count = int(parameters["profiles"])
    ood_count = int(parameters["ood_profiles"])
    observed_profile_indices: set[int] = set()
    for profile in profiles:
        index = int(profile["profile_index"])
        expected_seed = stable_seed(master_seed, "d3-profile", index)
        expected_domain = "OOD" if index >= profile_count - ood_count else "ID"
        expected_values = _expected_profile_values(
            contract, expected_seed, expected_domain
        )
        expected_hash = sha256_json(expected_values)
        expected_id = f"D3P{index:04d}-{expected_hash[:10]}"
        observed_profile_indices.add(index)
        profile_contract_valid = profile_contract_valid and all(
            (
                0 <= index < profile_count,
                int(profile["profile_seed"]) == expected_seed,
                profile["domain"] == expected_domain,
                profile["plant_profile_id"] == expected_id,
                profile["profile_config_hash"] == expected_hash,
                all(
                    _close(profile[name], expected_values[name])
                    for name in expected_values
                ),
            )
        )
    profile_contract_valid = profile_contract_valid and (
        observed_profile_indices == set(range(profile_count))
        and len(profiles) == profile_count
    )
    for row in trajectories:
        profile = profile_by_id.get(row["plant_profile_id"])
        if profile is None:
            trajectory_sampling_valid = False
            continue
        index = int(row["trajectory_index"])
        expected_degradation_seed = stable_seed(
            master_seed, "d3-degradation", row["plant_profile_id"], index
        )
        expected_censor_seed = stable_seed(
            master_seed, "d3-censor", row["plant_profile_id"], index
        )
        expected_noise_seed = stable_seed(
            master_seed, "d3-noise", row["plant_profile_id"], index
        )
        expected_acceleration = _rotated_choice(
            accelerations, expected_degradation_seed, index
        )
        expected_horizon = _rotated_choice(horizons, expected_censor_seed, index)
        trajectory_sampling_valid = trajectory_sampling_valid and all(
            (
                int(row["degradation_seed"]) == expected_degradation_seed,
                int(row["censor_seed"]) == expected_censor_seed,
                int(row["noise_seed"]) == expected_noise_seed,
                _close(row["degradation_acceleration_factor"], expected_acceleration),
                _close(row["requested_censor_horizon_equivalent_h"], expected_horizon),
                _close(
                    row["accelerated_fouling_rate_per_h"],
                    float(row["base_fouling_rate_per_h"]) * expected_acceleration,
                ),
                row["profile_config_hash"] == profile["profile_config_hash"],
                row["model_version"] == manifest["model_version"],
                row["contract_version"] == manifest["contract_version"],
                row["generator_version"] == manifest["generator_version"],
            )
        )
        event = row["event_observed"] == "1"
        outcome_metadata_valid = outcome_metadata_valid and (
            (event and row["outcome_type"] == "EVENT")
            or (not event and row["outcome_type"] == "RIGHT_CENSORED")
        )
        if event:
            outcome_metadata_valid = outcome_metadata_valid and all(
                (
                    bool(row["eol_time_sim_s"]),
                    bool(row["eol_time_equivalent_s"]),
                    bool(row["eol_cause"]),
                    bool(row["eol_cause_mask"]),
                    not row["censor_time_sim_s"],
                    not row["censor_time_equivalent_s"],
                    _close(row["observation_end_sim_s"], row["eol_time_sim_s"]),
                )
            )
        else:
            outcome_metadata_valid = outcome_metadata_valid and all(
                (
                    not row["eol_time_sim_s"],
                    not row["eol_time_equivalent_s"],
                    not row["eol_cause"],
                    not row["eol_cause_mask"],
                    bool(row["censor_time_sim_s"]),
                    bool(row["censor_time_equivalent_s"]),
                    _close(row["observation_end_sim_s"], row["censor_time_sim_s"]),
                )
            )
        outcome_metadata_valid = outcome_metadata_valid and (
            float(row["observation_end_sim_s"])
            <= float(row["administrative_censor_time_sim_s"]) + EPSILON
            and _close(
                row["observation_end_equivalent_s"],
                float(row["observation_end_sim_s"])
                * float(row["degradation_acceleration_factor"]),
            )
        )
    audit.check(
        "profile_sampling_and_hashes",
        profile_contract_valid,
        "profile IDs, domains, values and hashes reproduce from seed and ID/OOD ranges",
    )
    audit.check(
        "independent_degradation_and_censor_sampling",
        trajectory_sampling_valid,
        "dedicated seeds deterministically reproduce acceleration, horizon and noise",
    )
    audit.check(
        "trajectory_outcome_metadata",
        outcome_metadata_valid,
        "event and administrative-censor metadata are mutually exclusive and unit-consistent",
    )

    evidence_by_trajectory: dict[str, list[dict[str, str]]] = defaultdict(list)
    evidence_valid = True
    required_persistence = {
        "FOULING_LIMIT": 0.0,
        "STEAM_SATURATION": float(contract["eol"]["steam_persistence_sim_s"]),
        "ENERGY_INTENSITY": float(contract["eol"]["energy_persistence_sim_s"]),
        "LOSS_OF_FORWARD": float(contract["eol"]["divert_persistence_sim_s"]),
    }
    threshold_check = {
        "FOULING_LIMIT": lambda value: value
        >= float(contract["eol"]["fouling_index_threshold"]),
        "STEAM_SATURATION": lambda value: value
        >= float(contract["eol"]["steam_valve_threshold"]),
        "ENERGY_INTENSITY": lambda value: value
        >= float(contract["eol"]["energy_intensity_ratio_threshold"]),
        "LOSS_OF_FORWARD": lambda value: value
        <= float(contract["eol"]["divert_position_threshold"]),
    }
    seen_evidence: set[tuple[str, str]] = set()
    for item in evidence_rows:
        trajectory = trajectory_by_id.get(item["trajectory_id"])
        condition = item["condition"]
        key = (item["trajectory_id"], condition)
        if trajectory is None or condition not in required_persistence or key in seen_evidence:
            evidence_valid = False
            continue
        seen_evidence.add(key)
        start = float(item["persistence_start_sim_s"])
        trigger = float(item["trigger_time_sim_s"])
        value = float(item["trigger_value"])
        acceleration = float(trajectory["degradation_acceleration_factor"])
        evidence_valid = evidence_valid and all(
            (
                trigger + EPSILON >= start + required_persistence[condition],
                trigger <= float(trajectory["observation_end_sim_s"]) + EPSILON,
                _close(item["trigger_time_equivalent_s"], trigger * acceleration),
                threshold_check[condition](value),
            )
        )
        evidence_by_trajectory[item["trajectory_id"]].append(item)
    for trajectory in trajectories:
        items = evidence_by_trajectory[trajectory["trajectory_id"]]
        event = trajectory["event_observed"] == "1"
        if event:
            if not items:
                evidence_valid = False
                continue
            earliest = min(float(item["trigger_time_sim_s"]) for item in items)
            earliest_conditions = {
                item["condition"]
                for item in items
                if _close(item["trigger_time_sim_s"], earliest)
            }
            evidence_valid = evidence_valid and (
                _close(earliest, trajectory["eol_time_sim_s"])
                and trajectory["eol_cause"] in earliest_conditions
                and set(trajectory["eol_cause_mask"].split("|")) == earliest_conditions
            )
        else:
            evidence_valid = evidence_valid and not items
    row_alignment_valid = True
    label_semantics_valid = True
    health_valid = True
    row_counts: Counter[str] = Counter()
    last_time: dict[str, float] = {}
    last_fouling: dict[str, float] = {}
    last_armed: dict[str, int] = {}
    last_label_by_trajectory: dict[str, dict[str, str]] = {}
    eol_trace_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reported_armed: dict[str, list[int]] = defaultdict(list)
    with (
        (args.dataset / "signals.csv").open(newline="", encoding="utf-8") as signal_handle,
        (args.dataset / "oracle_labels.csv").open(newline="", encoding="utf-8") as label_handle,
    ):
        signals = csv.DictReader(signal_handle)
        labels = csv.DictReader(label_handle)
        while True:
            signal = next(signals, None)
            label = next(labels, None)
            if signal is None or label is None:
                row_alignment_valid = row_alignment_valid and signal is None and label is None
                break
            keys = ("trajectory_id", "time_start_sim_s", "time_sim_s")
            row_alignment_valid = row_alignment_valid and all(
                signal[key] == label[key] for key in keys
            )
            trajectory_id = signal["trajectory_id"]
            trajectory = trajectory_by_id.get(trajectory_id)
            if trajectory is None:
                row_alignment_valid = False
                continue
            time_start_s = float(signal["time_start_sim_s"])
            time_s = float(signal["time_sim_s"])
            step_dt_s = float(signal["step_dt_sim_s"])
            expected_start_s = last_time.get(trajectory_id, 0.0)
            row_alignment_valid = row_alignment_valid and (
                time_s > time_start_s
                and _close(time_start_s, expected_start_s)
                and _close(step_dt_s, time_s - time_start_s)
            )
            last_time[trajectory_id] = time_s
            row_counts[trajectory_id] += 1
            acceleration = float(trajectory["degradation_acceleration_factor"])
            health = float(label["health_index"])
            fouling = float(label["fouling_index"])
            health_valid = health_valid and (
                -EPSILON <= fouling <= 1.0 + EPSILON
                and _close(health, 1.0 - fouling)
                and fouling + EPSILON >= last_fouling.get(trajectory_id, fouling)
                and _close(signal["operating_time_meter_h"], time_s / 3600.0)
                and int(label["maintenance_needed"])
                == int(fouling >= float(contract["maintenance_fouling_index"]))
            )
            last_fouling[trajectory_id] = fouling
            armed = int(label["eol_monitor_armed"])
            health_valid = health_valid and armed >= last_armed.get(trajectory_id, armed)
            last_armed[trajectory_id] = armed
            reported_armed[trajectory_id].append(armed)
            eol_trace_rows[trajectory_id].append(
                {
                    "time_start_sim_s": time_start_s,
                    "time_sim_s": time_s,
                    "step_dt_sim_s": step_dt_s,
                    "fouling_index": label["fouling_index"],
                    "steam_valve": signal["steam_valve"],
                    "fdv_actual_forward": label["fdv_actual_forward"],
                    "actual_energy_intensity_ratio": label[
                        "actual_energy_intensity_ratio"
                    ],
                }
            )
            event = trajectory["event_observed"] == "1"
            remaining_sim = max(0.0, float(trajectory["observation_end_sim_s"]) - time_s)
            remaining_equivalent = remaining_sim * acceleration
            label_semantics_valid = label_semantics_valid and all(
                (
                    label["event_observed"] == trajectory["event_observed"],
                    label["eol_cause"] == trajectory["eol_cause"],
                    label["eol_cause_mask"] == trajectory["eol_cause_mask"],
                    _close(label["time_to_event_or_censor_sim_s"], remaining_sim),
                    _close(
                        label["time_to_event_or_censor_equivalent_s"],
                        remaining_equivalent,
                    ),
                    int(label["terminal_event"])
                    == int(
                        event
                        and _close(time_s, trajectory["observation_end_sim_s"])
                    ),
                )
            )
            if event:
                label_semantics_valid = label_semantics_valid and all(
                    (
                        bool(label["rul_sim_s"]),
                        bool(label["rul_equivalent_s"]),
                        not label["rul_lower_bound_sim_s"],
                        not label["rul_lower_bound_equivalent_s"],
                        _close(label["rul_sim_s"], remaining_sim),
                        _close(label["rul_equivalent_s"], remaining_equivalent),
                    )
                )
            else:
                label_semantics_valid = label_semantics_valid and all(
                    (
                        not label["rul_sim_s"],
                        not label["rul_equivalent_s"],
                        bool(label["rul_lower_bound_sim_s"]),
                        bool(label["rul_lower_bound_equivalent_s"]),
                        _close(label["rul_lower_bound_sim_s"], remaining_sim),
                        _close(
                            label["rul_lower_bound_equivalent_s"],
                            remaining_equivalent,
                        ),
                    )
                )
            last_label_by_trajectory[trajectory_id] = label

    # Recompute the complete EOL state machine from the emitted, contiguous
    # trace.  eol_evidence.csv is treated only as a claim to compare against,
    # never as proof that its own persistence fields are correct.
    recomputed_eol_valid = evidence_valid
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        rows = eol_trace_rows.get(trajectory_id, [])
        try:
            expected_trace, expected_evidence, expected_outcome = evaluate_eol_trace(
                rows,
                float(trajectory["degradation_acceleration_factor"]),
                contract["eol"],
            )
        except (KeyError, TypeError, ValueError):
            recomputed_eol_valid = False
            continue
        expected_armed = [int(item["eol_monitor_armed"]) for item in expected_trace]
        recomputed_eol_valid = recomputed_eol_valid and (
            expected_armed == reported_armed.get(trajectory_id, [])
        )

        reported_items = {
            item["condition"]: item
            for item in evidence_by_trajectory.get(trajectory_id, [])
        }
        recomputed_eol_valid = recomputed_eol_valid and (
            set(reported_items) == set(expected_evidence)
        )
        for condition, expected in expected_evidence.items():
            reported = reported_items.get(condition)
            if reported is None:
                recomputed_eol_valid = False
                continue
            recomputed_eol_valid = recomputed_eol_valid and all(
                (
                    reported["condition"] == expected["condition"],
                    _close(
                        reported["persistence_start_sim_s"],
                        expected["persistence_start_sim_s"],
                    ),
                    _close(
                        reported["trigger_time_sim_s"],
                        expected["trigger_time_sim_s"],
                    ),
                    _close(
                        reported["trigger_time_equivalent_s"],
                        expected["trigger_time_equivalent_s"],
                    ),
                    _close(reported["trigger_value"], expected["trigger_value"]),
                )
            )

        event = trajectory["event_observed"] == "1"
        if event:
            recomputed_eol_valid = recomputed_eol_valid and (
                expected_outcome is not None
            )
            if expected_outcome is not None:
                recomputed_eol_valid = recomputed_eol_valid and all(
                    (
                        _close(
                            trajectory["eol_time_sim_s"],
                            expected_outcome["eol_time_sim_s"],
                        ),
                        _close(
                            trajectory["eol_time_equivalent_s"],
                            expected_outcome["eol_time_equivalent_s"],
                        ),
                        trajectory["eol_cause"]
                        == expected_outcome["eol_cause"],
                        trajectory["eol_cause_mask"]
                        == expected_outcome["eol_cause_mask"],
                    )
                )
        else:
            recomputed_eol_valid = recomputed_eol_valid and (
                expected_outcome is None
                and not expected_evidence
                and not reported_items
            )
    audit.check(
        "eol_persistence_and_first_cause",
        recomputed_eol_valid,
        "emitted trace independently reproduces continuity, arming, thresholds, persistence evidence and first-cause outcome",
    )

    terminal_valid = True
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        last = last_label_by_trajectory.get(trajectory_id)
        if last is None:
            terminal_valid = False
            continue
        terminal_valid = terminal_valid and (
            _close(last["time_sim_s"], trajectory["observation_end_sim_s"])
            and _close(last["time_to_event_or_censor_sim_s"], 0.0)
            and int(last["terminal_event"]) == int(trajectory["event_observed"])
            and row_counts[trajectory_id] == int(trajectory["row_count"])
        )
    audit.check(
        "signal_label_alignment_and_counts",
        row_alignment_valid
        and dict(row_counts)
        == {row["trajectory_id"]: int(row["row_count"]) for row in trajectories},
        "signal/label keys are aligned, monotone and match trajectory row counts",
    )
    audit.check(
        "health_and_unscaled_clock_invariants",
        health_valid,
        "health=1-fouling, production fouling is monotone, arm is sticky and the observable clock is unscaled simulator time",
    )
    audit.check(
        "right_censoring_and_rul_semantics",
        label_semantics_valid and terminal_valid,
        "censored RUL is null with a lower bound; observed RUL and terminal rows are exact",
    )
    manifest_counts = manifest["counts"]
    observed_counts = {
        "profiles": len(profiles),
        "trajectories": len(trajectories),
        "signal_rows": sum(row_counts.values()),
        "label_rows": sum(row_counts.values()),
        "eol_evidence_rows": len(evidence_rows),
        "events": sum(row["event_observed"] == "1" for row in trajectories),
        "right_censored": sum(row["event_observed"] == "0" for row in trajectories),
    }
    audit.check(
        "manifest_counts",
        all(manifest_counts.get(key) == value for key, value in observed_counts.items()),
        f"manifest={manifest_counts}, observed={observed_counts}",
    )
    split_outcomes: dict[str, set[str]] = defaultdict(set)
    split_by_trajectory = {row["trajectory_id"]: row["split"] for row in splits}
    for row in trajectories:
        split_outcomes[split_by_trajectory[row["trajectory_id"]]].add(row["outcome_type"])
    if any(len(values) < 2 for values in split_outcomes.values()):
        audit.warn(
            "At least one split lacks either an observed EOL or a right-censored trajectory; "
            "acceptable for smoke data but not for a full benchmark."
        )
    report = {
        "auditor_version": AUDITOR_VERSION,
        "status": "passed" if not audit.errors else "failed",
        "dataset_family": "D3-RUL",
        "dataset_version": manifest.get("dataset_version"),
        "model_version": manifest.get("model_version"),
        "checks": audit.checks,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "summary": {
            **observed_counts,
            "split_outcomes": {
                split: sorted(values) for split, values in sorted(split_outcomes.items())
            },
        },
        "provenance": {
            "audit_d3_rul_dataset.py": sha256_file(Path(__file__).resolve()),
            "d3_rul_common.py": sha256_file(BASE_DIR / "d3_rul_common.py"),
            "d3_rul_contract.json": sha256_file(CONTRACT_PATH),
            "dataset_manifest.json": sha256_file(manifest_path),
            "split_manifest.json": sha256_file(split_manifest_path),
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
        f"{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    print(
        f"D3 audit {report['status']}: {len(report['checks'])} checks, "
        f"{len(report['errors'])} failures; report={report_path}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
