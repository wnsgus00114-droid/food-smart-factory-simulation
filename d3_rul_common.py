#!/usr/bin/env python3
"""Shared, D1-independent helpers for the D3 scalar-fouling/RUL pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = BASE_DIR / "d3_rul_contract.json"
EPSILON = 1.0e-9


def load_d3_contract() -> dict[str, Any]:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if value.get("dataset_family") != "D3-RUL":
        raise ValueError("d3_rul_contract.json has the wrong dataset_family")
    base_ranges = value.get("profile_parameter_ranges")
    ood_ranges = value.get("ood_profile_parameter_ranges")
    if not isinstance(base_ranges, dict) or not base_ranges:
        raise ValueError("D3 profile_parameter_ranges must be a non-empty object")
    if not isinstance(ood_ranges, dict) or not ood_ranges:
        raise ValueError("D3 ood_profile_parameter_ranges must be a non-empty object")
    if not set(ood_ranges).issubset(base_ranges):
        raise ValueError("D3 OOD range names must be a subset of ID range names")
    for range_kind, ranges in (("ID", base_ranges), ("OOD", ood_ranges)):
        for name, bounds in ranges.items():
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(isinstance(item, bool) for item in bounds)
                or any(not isinstance(item, (int, float)) for item in bounds)
                or any(not math.isfinite(float(item)) for item in bounds)
                or float(bounds[0]) >= float(bounds[1])
            ):
                raise ValueError(
                    f"D3 {range_kind} range {name!r} must contain finite "
                    "increasing bounds"
                )
    for name, bounds in ood_ranges.items():
        id_low, id_high = (float(item) for item in base_ranges[name])
        ood_low, ood_high = (float(item) for item in bounds)
        if not (ood_high < id_low or ood_low > id_high):
            raise ValueError(
                f"D3 OOD range {name!r} must not overlap ID support"
            )
    return value


def model_version_is_compatible(version: str, contract: Mapping[str, Any]) -> bool:
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return major in {int(item) for item in contract["compatible_model_major_versions"]}


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv_header(handle: Any, fields: Sequence[str]) -> csv.DictWriter:
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    return writer


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def write_checksums(directory: Path, names: Sequence[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(names)]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(directory: Path) -> list[str]:
    checksum_path = directory / "checksums.sha256"
    if not checksum_path.exists():
        return [f"missing {checksum_path}"]
    failures: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        path = directory / name
        if not separator or not path.is_file():
            failures.append(f"invalid or missing checksum target: {name!r}")
        elif sha256_file(path) != digest:
            failures.append(f"checksum mismatch: {name}")
    return failures


def prepare_empty_output(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def feature_fields(contract: Mapping[str, Any], feature_set: str = "H1-age") -> list[str]:
    return list(contract["feature_sets"][feature_set])


def field_declaration(
    name: str, declared_type: str, nullable: bool, role: str
) -> dict[str, Any]:
    return {
        "name": name,
        "type": declared_type,
        "nullable": nullable,
        "role": role,
    }


def write_schema(
    path: Path,
    version: str,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    write_json(
        path,
        {
            "schema_version": version,
            "dataset_family": "D3-RUL",
            "tables": {
                name: {"fields": [dict(field) for field in fields]}
                for name, fields in tables.items()
            },
        },
    )


def csv_value_matches_type(value: str, declared: str, nullable: bool) -> bool:
    if value == "":
        return nullable
    try:
        if declared == "integer":
            integer = int(value)
            return str(integer) == value or float(value) == integer
        if declared == "number":
            return math.isfinite(float(value))
        if declared == "boolean":
            return value.lower() in {"true", "false", "0", "1"}
        return declared == "string"
    except ValueError:
        return False


def validate_declared_schema(path: Path, fields: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    names = [str(field["name"]) for field in fields]
    if read_csv_header(path) != names:
        return False, f"{path.name}: header differs from declared schema"
    declarations = {str(field["name"]): field for field in fields}
    for row_number, row in enumerate(read_csv_rows(path), start=2):
        for name in names:
            item = declarations[name]
            if not csv_value_matches_type(
                row[name], str(item["type"]), bool(item["nullable"])
            ):
                return False, f"{path.name}:{row_number}:{name} violates schema"
    return True, f"{path.name}: schema valid"


def select_sample_indices(
    rows: Sequence[Mapping[str, Any]], end_time_s: float, interval_s: float
) -> list[int]:
    """Select regular end-time samples and always retain the terminal row."""
    if interval_s <= 0.0:
        raise ValueError("sample interval must be positive")
    selected: list[int] = []
    next_sample = interval_s
    terminal_index: int | None = None
    for index, row in enumerate(rows):
        time_s = float(row["time_s"])
        if time_s > end_time_s + EPSILON:
            break
        terminal_index = index
        if time_s + EPSILON >= next_sample:
            selected.append(index)
            while next_sample <= time_s + EPSILON:
                next_sample += interval_s
    if terminal_index is None:
        raise ValueError("trajectory has no row at or before its observation end")
    if not selected or selected[-1] != terminal_index:
        selected.append(terminal_index)
    return selected


def _condition_trigger(
    *,
    active: bool,
    required_s: float,
    dt_s: float,
    start_s: float,
    end_s: float,
    accumulated_s: float,
    onset_s: float | None,
) -> tuple[float, float | None, float | None]:
    if not active:
        return 0.0, None, None
    if onset_s is None:
        onset_s = start_s
    accumulated_s += dt_s
    trigger_s = end_s if accumulated_s + EPSILON >= required_s else None
    return accumulated_s, onset_s, trigger_s


def evaluate_eol_trace(
    rows: Sequence[Mapping[str, Any]],
    acceleration: float,
    eol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Evaluate composite EOL from an oracle trace.

    Each row must contain time bounds, fouling_index, steam_valve,
    fdv_actual_forward, and actual_energy_intensity_ratio (blank/None until the
    rolling reference window is available). Persistence is deliberately in
    simulator seconds; equivalent-age labels scale only the resulting clock.
    """
    if not math.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError("acceleration must be finite and positive")
    priority = [str(value) for value in eol["cause_priority"]]
    evidence: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    forward_accum = 0.0
    armed = False
    accum = {cause: 0.0 for cause in priority}
    onset: dict[str, float | None] = {cause: None for cause in priority}

    for row in rows:
        start_s = float(row["time_start_sim_s"])
        end_s = float(row["time_sim_s"])
        dt_s = float(row["step_dt_sim_s"])
        if end_s <= start_s or not math.isclose(end_s - start_s, dt_s, abs_tol=1e-7):
            raise ValueError("invalid time interval in EOL trace")
        actual_forward = float(row["fdv_actual_forward"]) >= float(
            eol["forward_arm_position"]
        )
        if end_s + EPSILON >= float(eol["minimum_monitor_time_sim_s"]) and actual_forward:
            forward_accum += dt_s
        else:
            forward_accum = 0.0
        if forward_accum + EPSILON >= float(eol["forward_arm_persistence_sim_s"]):
            armed = True

        energy_text = row.get("actual_energy_intensity_ratio", "")
        energy_ratio = None if energy_text in {"", None} else float(energy_text)
        active_by_cause = {
            "FOULING_LIMIT": (
                armed
                and end_s + EPSILON >= float(eol["minimum_monitor_time_sim_s"])
                and float(row["fouling_index"])
                >= float(eol["fouling_index_threshold"])
            ),
            "STEAM_SATURATION": (
                armed
                and float(row["steam_valve"])
                >= float(eol["steam_valve_threshold"])
            ),
            "ENERGY_INTENSITY": (
                armed
                and energy_ratio is not None
                and energy_ratio >= float(eol["energy_intensity_ratio_threshold"])
            ),
            "LOSS_OF_FORWARD": (
                armed
                and float(row["fdv_actual_forward"])
                <= float(eol["divert_position_threshold"])
            ),
        }
        required = {
            "FOULING_LIMIT": 0.0,
            "STEAM_SATURATION": float(eol["steam_persistence_sim_s"]),
            "ENERGY_INTENSITY": float(eol["energy_persistence_sim_s"]),
            "LOSS_OF_FORWARD": float(eol["divert_persistence_sim_s"]),
        }
        values = {
            "FOULING_LIMIT": float(row["fouling_index"]),
            "STEAM_SATURATION": float(row["steam_valve"]),
            "ENERGY_INTENSITY": energy_ratio,
            "LOSS_OF_FORWARD": float(row["fdv_actual_forward"]),
        }
        completed_now: list[str] = []
        for cause in priority:
            if cause in evidence:
                continue
            accumulated, started, trigger = _condition_trigger(
                active=active_by_cause[cause],
                required_s=required[cause],
                dt_s=dt_s,
                start_s=start_s,
                end_s=end_s,
                accumulated_s=accum[cause],
                onset_s=onset[cause],
            )
            accum[cause], onset[cause] = accumulated, started
            if trigger is not None:
                evidence[cause] = {
                    "condition": cause,
                    "persistence_start_sim_s": started,
                    "trigger_time_sim_s": trigger,
                    "trigger_time_equivalent_s": trigger * acceleration,
                    "trigger_value": values[cause],
                }
                completed_now.append(cause)
        trace.append(
            {
                "eol_monitor_armed": int(armed),
                "conditions_completed": tuple(completed_now),
            }
        )

    if not evidence:
        return trace, evidence, None
    earliest = min(float(item["trigger_time_sim_s"]) for item in evidence.values())
    causes = [
        cause
        for cause in priority
        if cause in evidence
        and math.isclose(
            float(evidence[cause]["trigger_time_sim_s"]), earliest, abs_tol=EPSILON
        )
    ]
    outcome = {
        "eol_time_sim_s": earliest,
        "eol_time_equivalent_s": earliest * acceleration,
        "eol_cause": causes[0],
        "eol_cause_mask": "|".join(causes),
    }
    return trace, evidence, outcome


def rolling_energy_intensity_ratios(
    actual_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    window_s: float,
) -> list[float | None]:
    if len(actual_rows) != len(reference_rows):
        raise ValueError("actual/reference rows have different lengths")
    history: deque[tuple[float, float, float, float, float]] = deque()
    actual_energy = actual_volume = reference_energy = reference_volume = 0.0
    ratios: list[float | None] = []
    for actual, reference in zip(actual_rows, reference_rows):
        end_s = float(actual["time_s"])
        if not math.isclose(end_s, float(reference["time_s"]), abs_tol=EPSILON):
            raise ValueError("actual/reference time grids differ")
        item = (
            end_s,
            float(actual["heat_energy_kwh"]) + float(actual["pump_energy_kwh"]),
            float(actual["routed_volume_l"]),
            float(reference["heat_energy_kwh"]) + float(reference["pump_energy_kwh"]),
            float(reference["routed_volume_l"]),
        )
        history.append(item)
        actual_energy += item[1]
        actual_volume += item[2]
        reference_energy += item[3]
        reference_volume += item[4]
        cutoff = end_s - window_s
        while history and history[0][0] <= cutoff + EPSILON:
            expired = history.popleft()
            actual_energy -= expired[1]
            actual_volume -= expired[2]
            reference_energy -= expired[3]
            reference_volume -= expired[4]
        if end_s + EPSILON < window_s or min(actual_volume, reference_volume) <= EPSILON:
            ratios.append(None)
            continue
        actual_intensity = actual_energy / actual_volume
        reference_intensity = reference_energy / reference_volume
        ratios.append(
            actual_intensity / reference_intensity
            if reference_intensity > EPSILON
            else None
        )
    return ratios
