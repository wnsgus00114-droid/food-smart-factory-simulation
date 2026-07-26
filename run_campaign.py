#!/usr/bin/env python3
"""Run a state-continuous HTST production/CIP/restart campaign.

The ordinary ``run_scenarios.py`` command intentionally keeps each scenario
independent.  This command uses the campaign API, records every carried state
at phase boundaries, and publishes the whole artifact set atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from model import ALARM_COLUMNS, HTSTConfig, HTSTSimulator, MODEL_VERSION, SCENARIO_NAMES
from provenance import source_provenance, staged_output_directory, write_checksums
from run_scenarios import field_role, field_type, write_csv, write_json


HERE = Path(__file__).resolve().parent
CAMPAIGN_ARTIFACT_VERSION = "1.0.0"
CAMPAIGN_SCHEMA_VERSION = "1.0.0"
TRANSITION_POLICY_VERSION = "1.0.0"
TRANSITION_POLICY = {
    "state": "carry",
    "balance_tank": "isolate_during_cip",
    "fdv": "command_divert_without_position_jump",
    "pi_integral": "reset_at_phase_boundary",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_CAMPAIGN_CONTEXT_FIELDS = {
    "campaign_id",
    "phase_id",
    "phase_index",
    "phase_kind",
    "phase_run_id",
    "campaign_time_start_s",
    "campaign_time_s",
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def _validate_identifier(name: str, value: object) -> str:
    text = str(value)
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(
            f"{name} must match {_SAFE_ID.pattern!r}; received {text!r}"
        )
    return text


def _validated_spec(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("campaign spec must be a JSON object")
    allowed = {
        "campaign_id",
        "description",
        "config",
        "phases",
        "campaign_duration_s",
        "transition_policy",
        "transition_policy_version",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown campaign spec keys: {sorted(unknown)}")
    campaign_id = _validate_identifier("campaign_id", raw.get("campaign_id", "campaign"))
    description = str(raw.get("description", ""))
    config_values = raw.get("config", {})
    if not isinstance(config_values, Mapping):
        raise ValueError("config must be an object")
    try:
        config = HTSTConfig(**dict(config_values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid campaign config: {exc}") from exc
    phase_values = raw.get("phases")
    if not isinstance(phase_values, list) or not phase_values:
        raise ValueError("phases must be a non-empty array")
    phases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(phase_values):
        if not isinstance(item, Mapping):
            raise ValueError(f"phases[{index}] must be an object")
        extra = set(item) - {"phase_id", "scenario", "duration_s"}
        if extra:
            raise ValueError(f"unknown phases[{index}] keys: {sorted(extra)}")
        phase_id = _validate_identifier(
            f"phases[{index}].phase_id", item.get("phase_id", f"phase-{index:03d}")
        )
        if phase_id in seen:
            raise ValueError(f"duplicate phase_id: {phase_id}")
        seen.add(phase_id)
        scenario = str(item.get("scenario", ""))
        if scenario not in SCENARIO_NAMES:
            raise ValueError(f"unknown phases[{index}].scenario: {scenario!r}")
        duration_s = item.get("duration_s")
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise ValueError(f"phases[{index}].duration_s must be numeric")
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"phases[{index}].duration_s must be finite and positive")
        if duration_s / config.dt_s > config.maximum_steps:
            raise ValueError(f"phases[{index}] exceeds maximum_steps")
        phases.append(
            {"phase_id": phase_id, "scenario": scenario, "duration_s": duration_s}
        )
    campaign_duration_s = sum(float(item["duration_s"]) for item in phases)
    declared_duration = raw.get("campaign_duration_s")
    if declared_duration is not None:
        if isinstance(declared_duration, bool) or not isinstance(
            declared_duration, (int, float)
        ):
            raise ValueError("campaign_duration_s must be numeric")
        if not math.isclose(
            float(declared_duration), campaign_duration_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "campaign_duration_s must equal the sum of phase durations: "
                f"{campaign_duration_s}"
            )
    policy_version = str(
        raw.get("transition_policy_version", TRANSITION_POLICY_VERSION)
    )
    if policy_version != TRANSITION_POLICY_VERSION:
        raise ValueError(
            "unsupported transition_policy_version: "
            f"{policy_version!r}; expected {TRANSITION_POLICY_VERSION!r}"
        )
    transition_policy = raw.get("transition_policy", TRANSITION_POLICY)
    if not isinstance(transition_policy, Mapping):
        raise ValueError("transition_policy must be an object")
    transition_policy = dict(transition_policy)
    missing_policy = set(TRANSITION_POLICY) - set(transition_policy)
    extra_policy = set(transition_policy) - set(TRANSITION_POLICY)
    if missing_policy or extra_policy:
        raise ValueError(
            "transition_policy keys do not match the implemented policy: "
            f"missing={sorted(missing_policy)}, extra={sorted(extra_policy)}"
        )
    mismatched_policy = {
        name: (TRANSITION_POLICY[name], transition_policy[name])
        for name in TRANSITION_POLICY
        if transition_policy[name] != TRANSITION_POLICY[name]
    }
    if mismatched_policy:
        raise ValueError(
            "transition_policy requests unsupported behavior: "
            f"{mismatched_policy}"
        )
    return {
        "campaign_id": campaign_id,
        "description": description,
        "config": asdict(config),
        "phases": phases,
        "campaign_duration_s": campaign_duration_s,
        "transition_policy": dict(TRANSITION_POLICY),
        "transition_policy_version": TRANSITION_POLICY_VERSION,
    }


def _schema(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot build a campaign schema from no rows")
    names = list(rows[0])
    expected = set(names)
    fields: list[dict[str, str]] = []
    for name in names:
        role = "context" if name in _CAMPAIGN_CONTEXT_FIELDS else field_role(name)
        fields.append({"name": name, "type": field_type(rows[0][name]), "role": role})
    for index, row in enumerate(rows):
        if set(row) != expected or list(row) != names:
            raise ValueError(f"campaign row {index} has schema drift")
        for field in fields:
            observed = field_type(row[field["name"]])
            if observed != field["type"]:
                raise ValueError(
                    f"campaign row {index} field {field['name']!r} type drift: "
                    f"{observed} != {field['type']}"
                )
    return {
        "campaign_artifact_version": CAMPAIGN_ARTIFACT_VERSION,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "roles": {
            "context": "Identifiers, clocks, modes and phase provenance.",
            "control": "Known commands or actuator state.",
            "observable_signal": "Signals available from the simulated instrumentation boundary.",
            "observable_alarm": "Alarms derived from observable or explicit status inputs.",
            "oracle": "Privileged physical truth or injected-fault state.",
            "outcome": "Accumulated material, safety or quality outcome.",
        },
        "field_count": len(fields),
        "fields": fields,
    }


def _campaign_events(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    previous_alarms = {name: 0 for name in ALARM_COLUMNS}
    previous_mode: str | None = None
    previous_phase: str | None = None
    for row in rows:
        at = float(row["campaign_time_start_s"])
        phase_id = str(row["phase_id"])
        if phase_id != previous_phase:
            events.append(
                {
                    "campaign_time_s": at,
                    "phase_id": phase_id,
                    "event_type": "PHASE",
                    "event_name": str(row["scenario"]),
                    "state": "ENTERED",
                    "role": "context",
                }
            )
            previous_phase = phase_id
            previous_mode = None
        mode = str(row["plant_mode"])
        if mode != previous_mode:
            events.append(
                {
                    "campaign_time_s": at,
                    "phase_id": phase_id,
                    "event_type": "MODE",
                    "event_name": mode,
                    "state": "ENTERED",
                    "role": "context",
                }
            )
            previous_mode = mode
        for alarm in ALARM_COLUMNS:
            state = int(row.get(alarm, 0))
            if state != previous_alarms[alarm]:
                events.append(
                    {
                        "campaign_time_s": at,
                        "phase_id": phase_id,
                        "event_type": "ALARM",
                        "event_name": alarm,
                        "state": "ACTIVE" if state else "CLEARED",
                        "role": "oracle" if alarm == "alarm_unsafe_forward" else "observable_alarm",
                    }
                )
                previous_alarms[alarm] = state
    return events


def _sum(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows)


def _state_scalar(state: Mapping[str, object], name: str) -> float:
    scalars = state.get("scalars")
    if not isinstance(scalars, Mapping):
        raise ValueError("campaign state has no scalar mapping")
    return float(scalars[name])


def _state_soil(state: Mapping[str, object]) -> tuple[float, float]:
    soil = state.get("cip_soil")
    if not isinstance(soil, Mapping):
        raise ValueError("campaign state has no CIP-soil mapping")
    total = float(soil["protein_soil_g"]) + float(soil["mineral_soil_g"])
    residual = float(soil["residual_alkali_fraction"]) + float(
        soil["residual_acid_fraction"]
    )
    return total, residual


def _phase_summary(
    phase: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    if not rows:
        raise ValueError(f"phase {phase['phase_id']!r} produced no rows")
    before_hash = _sha256_json(before)
    after_hash = _sha256_json(after)
    _, final_residual = _state_soil(after)
    return {
        **dict(phase),
        "row_count": len(rows),
        "campaign_time_start_s": float(rows[0]["campaign_time_start_s"]),
        "campaign_time_end_s": float(rows[-1]["campaign_time_s"]),
        "executed_duration_s": (
            float(rows[-1]["campaign_time_s"])
            - float(rows[0]["campaign_time_start_s"])
        ),
        "state_before_sha256": before_hash,
        "state_after_sha256": after_hash,
        "initial_fouling_index": _state_scalar(before, "fouling_index"),
        "final_fouling_index": _state_scalar(after, "fouling_index"),
        "initial_surface_soil_g": _state_soil(before)[0],
        "final_surface_soil_g": _state_soil(after)[0],
        "final_residual_chemical_fraction": final_residual,
        "forward_l": _sum(rows, "forward_l"),
        "unsafe_forward_l": _sum(rows, "unsafe_forward_l"),
        "chemical_noncompliant_forward_l": _sum(
            rows, "chemical_noncompliant_forward_l"
        ),
        "dilution_noncompliant_forward_l": _sum(
            rows, "dilution_noncompliant_forward_l"
        ),
        "hygiene_noncompliant_forward_l": _sum(
            rows, "hygiene_noncompliant_forward_l"
        ),
        "quality_out_of_spec_l": _sum(rows, "quality_out_of_spec_l"),
        "maximum_forward_product_temp_c": max(
            (
                float(row["potential_product_temp_c"])
                for row in rows
                if float(row["forward_l"]) > 0.0
            ),
            default=None,
        ),
        "max_abs_mass_balance_error_l": max(
            abs(float(row["mass_balance_error_l"])) for row in rows
        ),
        "max_abs_post_fdv_balance_error_l": max(
            abs(float(row["post_fdv_balance_error_l"])) for row in rows
        ),
        # The legacy row name is retained for schema compatibility.  In the
        # campaign summary this is described conservatively as processed flow;
        # the reduced-order CIP loop does not resolve a separate return pipe.
        "cip_processed_l": _sum(rows, "cip_recirculated_l"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path)
    return parser


def run_from_spec(
    spec_path: Path, output: Path, initial_state_path: Path | None = None
) -> Path:
    spec = _validated_spec(_load_json(spec_path))
    config = HTSTConfig(**spec["config"])
    simulator = HTSTSimulator(config)
    if initial_state_path is not None:
        imported = _load_json(initial_state_path)
        if not isinstance(imported, Mapping):
            raise ValueError("initial state must be a JSON object")
        simulator.import_state(imported)
    initial_state = simulator.export_state()
    initial_state_hash = _sha256_json(initial_state)
    all_rows: list[dict[str, object]] = []
    phase_summaries: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    checkpoints: list[tuple[str, dict[str, object]]] = []

    for index, phase in enumerate(spec["phases"]):
        before = simulator.export_state()
        before_hash = _sha256_json(before)
        rows = simulator.run_phase(
            str(phase["scenario"]),
            duration_s=float(phase["duration_s"]),
            reset=False,
            campaign_id=str(spec["campaign_id"]),
            phase_id=str(phase["phase_id"]),
            phase_index=index,
        )
        after = simulator.export_state()
        after_hash = _sha256_json(after)
        checkpoints.append((f"{index:03d}-{phase['phase_id']}.json", after))
        phase_summaries.append(_phase_summary(phase, rows, before, after))
        transitions.append(
            {
                "phase_index": index,
                "phase_id": phase["phase_id"],
                "scenario": phase["scenario"],
                "campaign_time_start_s": float(rows[0]["campaign_time_start_s"]),
                "campaign_time_s": float(rows[-1]["campaign_time_s"]),
                "state_before_sha256": before_hash,
                "state_after_sha256": after_hash,
                "previous_state_matches": (
                    True
                    if index == 0
                    else before_hash == transitions[-1]["state_after_sha256"]
                ),
                "policy_version": TRANSITION_POLICY_VERSION,
            }
        )
        all_rows.extend(rows)

    final_state = simulator.export_state()
    final_state_hash = _sha256_json(final_state)
    schema = _schema(all_rows)
    events = _campaign_events(all_rows)
    source = source_provenance(
        {
            "model.py": HERE / "model.py",
            "process_components.py": HERE / "process_components.py",
            "provenance.py": HERE / "provenance.py",
            "run_campaign.py": HERE / "run_campaign.py",
            "run_scenarios.py": HERE / "run_scenarios.py",
        }
    )
    initial_tank = initial_state["balance_tank"]
    final_tank = final_state["balance_tank"]
    initial_return = initial_state["pending_return_stream"]
    final_return = final_state["pending_return_stream"]
    if not all(
        isinstance(item, Mapping)
        for item in (initial_tank, final_tank, initial_return, final_return)
    ):
        raise ValueError("campaign state inventory mappings are invalid")
    total_fresh_feed_l = _sum(all_rows, "fresh_feed_l")
    total_forward_l = _sum(all_rows, "forward_l")
    total_transition_drained_l = _sum(all_rows, "transition_drained_l")
    external_volume_balance_error_l = (
        total_fresh_feed_l
        - total_forward_l
        - total_transition_drained_l
        - (float(final_tank["volume_l"]) - float(initial_tank["volume_l"]))
        - (float(final_return["volume_l"]) - float(initial_return["volume_l"]))
    )
    campaign_time_start_s = float(all_rows[0]["campaign_time_start_s"])
    campaign_time_end_s = float(all_rows[-1]["campaign_time_s"])
    summary = {
        "model_status": "unvalidated engineering research surrogate",
        "model_version": MODEL_VERSION,
        "campaign_artifact_version": CAMPAIGN_ARTIFACT_VERSION,
        "campaign_id": spec["campaign_id"],
        "campaign_time_start_s": campaign_time_start_s,
        "campaign_time_end_s": campaign_time_end_s,
        "duration_s": campaign_time_end_s - campaign_time_start_s,
        "row_count": len(all_rows),
        "phase_count": len(phase_summaries),
        "phases": phase_summaries,
        "totals": {
            "fresh_feed_l": total_fresh_feed_l,
            "forward_l": total_forward_l,
            "unsafe_forward_l": _sum(all_rows, "unsafe_forward_l"),
            "chemical_noncompliant_forward_l": _sum(
                all_rows, "chemical_noncompliant_forward_l"
            ),
            "dilution_noncompliant_forward_l": _sum(
                all_rows, "dilution_noncompliant_forward_l"
            ),
            "hygiene_noncompliant_forward_l": _sum(
                all_rows, "hygiene_noncompliant_forward_l"
            ),
            "quality_out_of_spec_l": _sum(all_rows, "quality_out_of_spec_l"),
            "transition_drained_l": total_transition_drained_l,
            "transition_added_l": _sum(all_rows, "transition_added_l"),
            "external_volume_balance_error_l": external_volume_balance_error_l,
            "max_abs_mass_balance_error_l": max(
                abs(float(row["mass_balance_error_l"])) for row in all_rows
            ),
            "max_abs_post_fdv_balance_error_l": max(
                abs(float(row["post_fdv_balance_error_l"])) for row in all_rows
            ),
        },
        "initial_state_sha256": initial_state_hash,
        "final_state_sha256": final_state_hash,
        "assurance_note": (
            "Campaign continuity and conservation are software assertions for an "
            "unvalidated surrogate, not plant hygiene or regulatory validation."
        ),
    }

    output = output.expanduser().resolve()
    with staged_output_directory(output) as staging:
        generated: list[Path] = []
        campaign_path = staging / "campaign.csv"
        write_csv(campaign_path, all_rows)
        generated.append(campaign_path)
        events_path = staging / "event_log.csv"
        write_csv(events_path, events)
        generated.append(events_path)
        spec_output = staging / "campaign_spec.json"
        write_json(spec_output, spec)
        generated.append(spec_output)
        summary_path = staging / "campaign_summary.json"
        write_json(summary_path, summary)
        generated.append(summary_path)
        transition_path = staging / "transition_log.json"
        write_json(transition_path, transitions)
        generated.append(transition_path)
        initial_path = staging / "initial_state.json"
        write_json(initial_path, initial_state)
        generated.append(initial_path)
        final_path = staging / "final_state.json"
        write_json(final_path, final_state)
        generated.append(final_path)
        checkpoint_dir = staging / "checkpoints"
        for name, state in checkpoints:
            path = checkpoint_dir / name
            write_json(path, state)
            generated.append(path)
        schema_path = staging / "schema.json"
        write_json(schema_path, schema)
        generated.append(schema_path)
        manifest_path = staging / "run_manifest.json"
        write_json(
            manifest_path,
            {
                "model_status": "unvalidated engineering research surrogate",
                "model_version": MODEL_VERSION,
                "campaign_artifact_version": CAMPAIGN_ARTIFACT_VERSION,
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": spec["campaign_id"],
                "canonical_spec_sha256": _sha256_json(spec),
                "initial_state_sha256": initial_state_hash,
                "final_state_sha256": final_state_hash,
                "ordered_phase_ids": [item["phase_id"] for item in spec["phases"]],
                "transition_policy_version": TRANSITION_POLICY_VERSION,
                "python_version": platform.python_version(),
                "source_provenance": source,
                "artifacts": [path.relative_to(staging).as_posix() for path in generated],
                "reproduce": {
                    "working_directory": "directory containing run_campaign.py",
                    "command_template": (
                        "python3 run_campaign.py "
                        "--spec <artifact_dir>/campaign_spec.json "
                        "--initial-state <artifact_dir>/initial_state.json "
                        "--output <new_output_dir>"
                    ),
                    "placeholders": {
                        "artifact_dir": "directory containing this manifest",
                        "new_output_dir": "an absent or replaceable output directory",
                    },
                },
            },
        )
        generated.append(manifest_path)
        write_checksums(staging, generated)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = run_from_spec(args.spec, args.output, args.initial_state)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"campaign configuration error: {exc}") from exc
    summary = _load_json(output / "campaign_summary.json")
    print(
        f"campaign={summary['campaign_id']} phases={summary['phase_count']} "
        f"rows={summary['row_count']} duration_s={summary['duration_s']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
