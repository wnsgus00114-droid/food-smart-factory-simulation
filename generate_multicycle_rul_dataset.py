#!/usr/bin/env python3
"""Generate auditable recurrent-event HTST lifecycle/RUL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Mapping, Sequence

from lifecycle import (
    EVENT_FIELDS,
    LIFECYCLE_MODEL_VERSION,
    MODEL_STATUS,
    SERVICE_INTERVAL_FIELDS,
    TRACE_FIELDS,
    PolicyValidationError,
    load_maintenance_policy,
    rows_have_exact_fields,
    simulate_lifecycle,
)
from provenance import source_provenance, staged_output_directory, write_checksums


HERE = Path(__file__).resolve().parent
MULTICYCLE_ARTIFACT_VERSION = "1.0.0"
MULTICYCLE_SCHEMA_VERSION = "1.0.0"

TRACE_INTEGER_FIELDS = {"seed", "cycle", "failures_in_cycle", "renewal_count"}
TRACE_NUMBER_FIELDS = {
    "cycle_end_time_h",
    "virtual_age_before_production_h",
    "damage_before_production",
    "fouling_before_production",
    "virtual_age_before_cip_h",
    "damage_before_cip",
    "fouling_before_cip",
    "damage_after_cip",
    "fouling_after_cip",
    "virtual_age_h",
    "irreversible_damage",
    "reversible_fouling",
    "weibull_median_rul_h",
    "damage_limit_rul_h",
    "fouling_limit_rul_h",
    "conditional_median_next_event_rul_h",
}
TRACE_NULLABLE_FIELDS = {"damage_limit_rul_h", "fouling_limit_rul_h"}

EVENT_INTEGER_FIELDS = {"cycle", "event_observed"}
EVENT_NUMBER_FIELDS = {
    "time_h",
    "virtual_age_before_h",
    "virtual_age_after_h",
    "damage_before",
    "damage_after",
    "fouling_before",
    "fouling_after",
    "virtual_age_factor_q",
    "damage_factor",
    "fouling_factor",
}

INTERVAL_INTEGER_FIELDS = {
    "interval_index",
    "generation",
    "start_cycle",
    "end_cycle",
    "event_observed",
    "is_last_interval",
}
INTERVAL_NUMBER_FIELDS = {
    "start_time_h",
    "end_time_h",
    "duration_h",
    "exact_rul_h",
    "rul_lower_bound_h",
    "start_virtual_age_h",
    "end_virtual_age_h",
    "start_damage",
    "end_damage",
    "start_fouling",
    "end_fouling",
}
INTERVAL_NULLABLE_FIELDS = {"exact_rul_h", "rul_lower_bound_h"}


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 2**63 - 1:
        raise argparse.ArgumentTypeError("must be in 0..2^63-1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=HERE / "maintenance_policy.json",
        help="strict maintenance policy JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "multicycle_rul_results",
        help="atomic output directory",
    )
    parser.add_argument(
        "--cycles",
        type=_positive_integer,
        default=None,
        help="override policy default cycle count",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help="override policy deterministic seed",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_rows(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    *,
    nullable_fields: set[str],
) -> None:
    if not rows_have_exact_fields(rows, fields):
        raise ValueError(f"row/schema column drift for fields {list(fields)}")
    for row_index, row in enumerate(rows):
        for name, value in row.items():
            if value is None:
                if name not in nullable_fields:
                    raise ValueError(
                        f"unexpected null at row {row_index}, field {name!r}"
                    )
                continue
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"non-finite number at row {row_index}, field {name!r}"
                )


def _role(table: str, name: str) -> str:
    if name in {
        "lifecycle_model_version",
        "run_id",
        "policy_id",
        "policy_hash",
        "seed",
        "cycle",
        "cycle_end_time_h",
        "event_id",
        "time_h",
        "asset_id",
        "component_type",
        "failure_mode",
        "service_interval_id",
        "interval_index",
        "generation",
        "start_time_h",
        "end_time_h",
        "start_cycle",
        "end_cycle",
    }:
        return "context"
    if name in {"cip_kind", "scheduled_action", "maintenance_action", "start_action"}:
        return "control"
    if table == "trace.csv" and name in {
        "conditional_median_next_event_rul_h",
        "median_competing_cause",
    }:
        return "outcome"
    if table == "service_intervals.csv" and name in {
        "terminal_event_type",
        "event_observed",
        "failure_cause",
        "exact_rul_h",
        "rul_lower_bound_h",
        "is_last_interval",
    }:
        return "outcome"
    if table == "events.csv" and name in {
        "event_type",
        "trigger_stage",
        "failure_cause",
        "event_observed",
    }:
        return "outcome"
    return "oracle"


def _field_schema(
    table: str,
    fields: Sequence[str],
    *,
    integer_fields: set[str],
    number_fields: set[str],
    nullable_fields: set[str],
) -> list[dict[str, object]]:
    declared: list[dict[str, object]] = []
    for name in fields:
        field_type = (
            "integer"
            if name in integer_fields
            else "number"
            if name in number_fields
            else "string"
        )
        declared.append(
            {
                "name": name,
                "type": field_type,
                "role": _role(table, name),
                "nullable": name in nullable_fields,
            }
        )
    return declared


def _schema() -> dict[str, object]:
    return {
        "schema_version": MULTICYCLE_SCHEMA_VERSION,
        "artifact_version": MULTICYCLE_ARTIFACT_VERSION,
        "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
        "roles": {
            "context": "Identity, time, configuration or grouping metadata.",
            "control": "CIP or maintenance action selected by policy.",
            "oracle": "Synthetic internal condition; forbidden as an online ML input.",
            "outcome": "Failure, censoring or RUL result; label/evaluation only.",
        },
        "tables": {
            "trace.csv": {
                "field_count": len(TRACE_FIELDS),
                "fields": _field_schema(
                    "trace.csv",
                    TRACE_FIELDS,
                    integer_fields=TRACE_INTEGER_FIELDS,
                    number_fields=TRACE_NUMBER_FIELDS,
                    nullable_fields=TRACE_NULLABLE_FIELDS,
                ),
            },
            "events.csv": {
                "field_count": len(EVENT_FIELDS),
                "fields": _field_schema(
                    "events.csv",
                    EVENT_FIELDS,
                    integer_fields=EVENT_INTEGER_FIELDS,
                    number_fields=EVENT_NUMBER_FIELDS,
                    nullable_fields=set(),
                ),
            },
            "service_intervals.csv": {
                "field_count": len(SERVICE_INTERVAL_FIELDS),
                "fields": _field_schema(
                    "service_intervals.csv",
                    SERVICE_INTERVAL_FIELDS,
                    integer_fields=INTERVAL_INTEGER_FIELDS,
                    number_fields=INTERVAL_NUMBER_FIELDS,
                    nullable_fields=INTERVAL_NULLABLE_FIELDS,
                ),
            },
        },
        "censoring_contract": {
            "observed_failure": "event_observed=1, exact_rul_h populated, rul_lower_bound_h null",
            "right_censored": "event_observed=0, exact_rul_h null, rul_lower_bound_h=observed interval duration",
            "last_interval": "the final interval for every asset is administratively right-censored",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_maintenance_policy(args.policy.expanduser().resolve())
        cycles = policy.default_cycles if args.cycles is None else args.cycles
        seed = policy.seed if args.seed is None else args.seed
        result = simulate_lifecycle(policy, cycles=cycles, seed=seed)
        _validate_rows(
            result.trace,
            TRACE_FIELDS,
            nullable_fields=TRACE_NULLABLE_FIELDS,
        )
        _validate_rows(result.events, EVENT_FIELDS, nullable_fields=set())
        _validate_rows(
            result.service_intervals,
            SERVICE_INTERVAL_FIELDS,
            nullable_fields=INTERVAL_NULLABLE_FIELDS,
        )
    except (PolicyValidationError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"multi-cycle lifecycle configuration error: {exc}") from exc

    output = args.output.expanduser().resolve()
    source = source_provenance(
        {
            "generate_multicycle_rul_dataset.py": HERE
            / "generate_multicycle_rul_dataset.py",
            "lifecycle.py": HERE / "lifecycle.py",
            "provenance.py": HERE / "provenance.py",
        }
    )
    artifact_names = [
        "trace.csv",
        "events.csv",
        "service_intervals.csv",
        "summary.json",
        "schema.json",
        "run_manifest.json",
    ]
    with staged_output_directory(output) as staging:
        trace_path = staging / "trace.csv"
        event_path = staging / "events.csv"
        intervals_path = staging / "service_intervals.csv"
        summary_path = staging / "summary.json"
        schema_path = staging / "schema.json"
        manifest_path = staging / "run_manifest.json"
        _write_csv(trace_path, result.trace, TRACE_FIELDS)
        _write_csv(event_path, result.events, EVENT_FIELDS)
        _write_csv(
            intervals_path,
            result.service_intervals,
            SERVICE_INTERVAL_FIELDS,
        )
        _write_json(summary_path, result.summary)
        _write_json(schema_path, _schema())
        _write_json(
            manifest_path,
            {
                "model_status": MODEL_STATUS,
                "artifact_version": MULTICYCLE_ARTIFACT_VERSION,
                "schema_version": MULTICYCLE_SCHEMA_VERSION,
                "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
                "python_version": platform.python_version(),
                "run_id": result.run_id,
                "policy_id": policy.policy_id,
                "policy_hash": result.policy_hash,
                "seed": seed,
                "cycles": cycles,
                "canonical_policy": policy.to_dict(),
                "source_provenance": source,
                "artifacts": artifact_names,
                "reproduce": (
                    "python3 generate_multicycle_rul_dataset.py "
                    "--policy <maintenance_policy.json> --output <output_dir> "
                    f"--cycles {cycles} --seed {seed}"
                ),
                "counts": result.summary["counts"],
            },
        )
        write_checksums(
            staging,
            [
                trace_path,
                event_path,
                intervals_path,
                summary_path,
                schema_path,
                manifest_path,
            ],
        )

    print(
        f"Generated multi-cycle lifecycle artifacts: cycles={cycles}, "
        f"failures={result.summary['counts']['failure_events']}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
