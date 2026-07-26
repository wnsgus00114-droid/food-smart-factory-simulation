#!/usr/bin/env python3
"""Synthetic HTST sensor dynamics, calibration, and uncertainty contracts.

The implementation follows the NIST TN 1297 convention of expressing input
uncertainty components as standard uncertainties, combining independent
components by root-sum-of-squares, and reporting expanded uncertainty as
``U = k * u_c``.  Nothing in this module establishes metrological traceability:
the bundled catalog and every record produced here are explicitly marked
``synthetic_reference``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CATALOG_PATH = Path(__file__).with_name("sensor_catalog.json")
SYNTHETIC_STATUS = "synthetic_reference"
STATE_SCHEMA_VERSION = "1.0.0"
CALIBRATION_SCHEMA_VERSION = "1.0.0"


class SensorValidationError(ValueError):
    """Raised when catalog, state, or calibration input is invalid."""


def _reject_json_constant(value: str) -> None:
    raise SensorValidationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SensorValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity tokens."""
    if not isinstance(text, str):
        raise TypeError("JSON input must be text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except SensorValidationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise SensorValidationError(f"invalid JSON: {exc}") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SensorValidationError(f"value is not strict JSON data: {exc}") from exc


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], context: str
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise SensorValidationError(
            f"{context} keys mismatch; missing={missing}, extra={extra}"
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SensorValidationError(f"{context} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise SensorValidationError(f"{context} contains a control character")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensorValidationError(f"{context} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise SensorValidationError(f"{context} must be finite")
    return converted


def _nonnegative(value: Any, context: str) -> float:
    converted = _finite_number(value, context)
    if converted < 0.0:
        raise SensorValidationError(f"{context} must be non-negative")
    return converted


def _positive(value: Any, context: str) -> float:
    converted = _finite_number(value, context)
    if converted <= 0.0:
        raise SensorValidationError(f"{context} must be positive")
    return converted


def _utc_datetime(value: datetime | str, context: str) -> datetime:
    if isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise SensorValidationError(f"{context} is not an ISO-8601 timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise SensorValidationError(f"{context} must be datetime or ISO-8601 text")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SensorValidationError(f"{context} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    status: str
    source_type: str
    source_id: str
    note: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], context: str, source_ids: set[str]
    ) -> "ProvenanceEntry":
        _require_exact_keys(
            value, {"status", "source_type", "source_id", "note"}, context
        )
        status = _nonempty_string(value["status"], f"{context}.status")
        if status != SYNTHETIC_STATUS:
            raise SensorValidationError(
                f"{context}.status must be {SYNTHETIC_STATUS!r}"
            )
        source_id = _nonempty_string(value["source_id"], f"{context}.source_id")
        if source_id not in source_ids:
            raise SensorValidationError(f"{context} references unknown source {source_id!r}")
        return cls(
            status=status,
            source_type=_nonempty_string(
                value["source_type"], f"{context}.source_type"
            ),
            source_id=source_id,
            note=_nonempty_string(value["note"], f"{context}.note"),
        )


@dataclass(frozen=True, slots=True)
class SensorSpec:
    tag: str
    status: str
    measurand: str
    unit: str
    minimum: float
    maximum: float
    lag_tau_s: float
    gain: float
    offset: float
    drift_per_hour: float
    hysteresis: float
    noise_std: float
    quantization: float
    dropout_probability: float
    hold_last: bool
    calibration_interval_days: float
    coverage_factor: float
    uncertainty_components: tuple[tuple[str, float], ...]
    guard_direction: str
    critical_limit: float
    parameter_provenance: tuple[tuple[str, ProvenanceEntry], ...]

    @property
    def uncertainty_component_map(self) -> dict[str, float]:
        return dict(self.uncertainty_components)

    @property
    def fingerprint(self) -> str:
        payload = {
            "tag": self.tag,
            "status": self.status,
            "measurand": self.measurand,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "chain": {
                "lag_tau_s": self.lag_tau_s,
                "gain": self.gain,
                "offset": self.offset,
                "drift_per_hour": self.drift_per_hour,
                "hysteresis": self.hysteresis,
                "noise_std": self.noise_std,
                "quantization": self.quantization,
                "dropout_probability": self.dropout_probability,
                "hold_last": self.hold_last,
            },
            "calibration_interval_days": self.calibration_interval_days,
            "coverage_factor": self.coverage_factor,
            "uncertainty_components": dict(self.uncertainty_components),
            "guard_direction": self.guard_direction,
            "critical_limit": self.critical_limit,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SensorCatalog:
    schema_version: str
    catalog_status: str
    claim_boundary: str
    sources: tuple[dict[str, str], ...]
    sensors: tuple[SensorSpec, ...]

    def sensor(self, tag: str) -> SensorSpec:
        for specification in self.sensors:
            if specification.tag == tag:
                return specification
        raise KeyError(f"sensor tag not found: {tag}")


def _numeric_parameter_paths(sensor: Mapping[str, Any]) -> set[str]:
    paths = {
        "engineering_range.minimum",
        "engineering_range.maximum",
        "chain.lag_tau_s",
        "chain.gain",
        "chain.offset",
        "chain.drift_per_hour",
        "chain.hysteresis",
        "chain.noise_std",
        "chain.quantization",
        "chain.dropout_probability",
        "calibration.interval_days",
        "calibration.coverage_factor",
        "guard_band.critical_limit",
    }
    components = sensor["calibration"]["uncertainty_components"]
    paths.update(
        f"calibration.uncertainty_components.{name}" for name in components
    )
    return paths


def _parse_sensor(
    value: Mapping[str, Any], index: int, source_ids: set[str]
) -> SensorSpec:
    context = f"sensors[{index}]"
    _require_exact_keys(
        value,
        {
            "tag",
            "status",
            "measurand",
            "unit",
            "engineering_range",
            "chain",
            "calibration",
            "guard_band",
            "parameter_provenance",
        },
        context,
    )
    status = _nonempty_string(value["status"], f"{context}.status")
    if status != SYNTHETIC_STATUS:
        raise SensorValidationError(
            f"{context}.status must be {SYNTHETIC_STATUS!r}"
        )

    engineering_range = value["engineering_range"]
    if not isinstance(engineering_range, dict):
        raise SensorValidationError(f"{context}.engineering_range must be an object")
    _require_exact_keys(
        engineering_range, {"minimum", "maximum"}, f"{context}.engineering_range"
    )
    minimum = _finite_number(
        engineering_range["minimum"], f"{context}.engineering_range.minimum"
    )
    maximum = _finite_number(
        engineering_range["maximum"], f"{context}.engineering_range.maximum"
    )
    if minimum >= maximum:
        raise SensorValidationError(f"{context} engineering range is not increasing")

    chain = value["chain"]
    if not isinstance(chain, dict):
        raise SensorValidationError(f"{context}.chain must be an object")
    _require_exact_keys(
        chain,
        {
            "lag_tau_s",
            "gain",
            "offset",
            "drift_per_hour",
            "hysteresis",
            "noise_std",
            "quantization",
            "dropout_probability",
            "hold_last",
        },
        f"{context}.chain",
    )
    lag_tau_s = _positive(chain["lag_tau_s"], f"{context}.chain.lag_tau_s")
    gain = _positive(chain["gain"], f"{context}.chain.gain")
    offset = _finite_number(chain["offset"], f"{context}.chain.offset")
    drift_per_hour = _finite_number(
        chain["drift_per_hour"], f"{context}.chain.drift_per_hour"
    )
    hysteresis = _nonnegative(
        chain["hysteresis"], f"{context}.chain.hysteresis"
    )
    noise_std = _nonnegative(chain["noise_std"], f"{context}.chain.noise_std")
    quantization = _positive(
        chain["quantization"], f"{context}.chain.quantization"
    )
    dropout_probability = _finite_number(
        chain["dropout_probability"], f"{context}.chain.dropout_probability"
    )
    if not 0.0 <= dropout_probability <= 1.0:
        raise SensorValidationError(
            f"{context}.chain.dropout_probability must be in [0, 1]"
        )
    if not isinstance(chain["hold_last"], bool):
        raise SensorValidationError(f"{context}.chain.hold_last must be boolean")

    calibration = value["calibration"]
    if not isinstance(calibration, dict):
        raise SensorValidationError(f"{context}.calibration must be an object")
    _require_exact_keys(
        calibration,
        {"interval_days", "coverage_factor", "uncertainty_components"},
        f"{context}.calibration",
    )
    interval_days = _positive(
        calibration["interval_days"], f"{context}.calibration.interval_days"
    )
    coverage_factor = _positive(
        calibration["coverage_factor"], f"{context}.calibration.coverage_factor"
    )
    component_data = calibration["uncertainty_components"]
    if not isinstance(component_data, dict) or not component_data:
        raise SensorValidationError(
            f"{context}.calibration.uncertainty_components must be a non-empty object"
        )
    components: list[tuple[str, float]] = []
    for name, component in sorted(component_data.items()):
        clean_name = _nonempty_string(
            name, f"{context}.calibration.uncertainty component name"
        )
        components.append(
            (
                clean_name,
                _nonnegative(
                    component,
                    f"{context}.calibration.uncertainty_components.{clean_name}",
                ),
            )
        )

    guard_band = value["guard_band"]
    if not isinstance(guard_band, dict):
        raise SensorValidationError(f"{context}.guard_band must be an object")
    _require_exact_keys(
        guard_band, {"direction", "critical_limit"}, f"{context}.guard_band"
    )
    direction = _nonempty_string(
        guard_band["direction"], f"{context}.guard_band.direction"
    )
    if direction not in {"lower", "upper"}:
        raise SensorValidationError(
            f"{context}.guard_band.direction must be 'lower' or 'upper'"
        )
    critical_limit = _finite_number(
        guard_band["critical_limit"], f"{context}.guard_band.critical_limit"
    )
    if not minimum <= critical_limit <= maximum:
        raise SensorValidationError(f"{context} critical limit is outside sensor range")

    provenance_data = value["parameter_provenance"]
    if not isinstance(provenance_data, dict):
        raise SensorValidationError(f"{context}.parameter_provenance must be an object")
    required_paths = _numeric_parameter_paths(value)
    if set(provenance_data) != required_paths:
        raise SensorValidationError(
            f"{context}.parameter_provenance must cover every numeric parameter; "
            f"missing={sorted(required_paths-set(provenance_data))}, "
            f"extra={sorted(set(provenance_data)-required_paths)}"
        )
    provenance: list[tuple[str, ProvenanceEntry]] = []
    for path, entry in sorted(provenance_data.items()):
        if not isinstance(entry, dict):
            raise SensorValidationError(
                f"{context}.parameter_provenance.{path} must be an object"
            )
        provenance.append(
            (
                path,
                ProvenanceEntry.from_mapping(
                    entry,
                    f"{context}.parameter_provenance.{path}",
                    source_ids,
                ),
            )
        )

    return SensorSpec(
        tag=_nonempty_string(value["tag"], f"{context}.tag"),
        status=status,
        measurand=_nonempty_string(value["measurand"], f"{context}.measurand"),
        unit=_nonempty_string(value["unit"], f"{context}.unit"),
        minimum=minimum,
        maximum=maximum,
        lag_tau_s=lag_tau_s,
        gain=gain,
        offset=offset,
        drift_per_hour=drift_per_hour,
        hysteresis=hysteresis,
        noise_std=noise_std,
        quantization=quantization,
        dropout_probability=dropout_probability,
        hold_last=chain["hold_last"],
        calibration_interval_days=interval_days,
        coverage_factor=coverage_factor,
        uncertainty_components=tuple(components),
        guard_direction=direction,
        critical_limit=critical_limit,
        parameter_provenance=tuple(provenance),
    )


def load_sensor_catalog(path: Path | str = CATALOG_PATH) -> SensorCatalog:
    """Load and strictly validate the synthetic sensor catalog."""
    catalog_path = Path(path)
    try:
        data = strict_json_loads(catalog_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SensorValidationError(f"cannot read sensor catalog: {catalog_path}") from exc
    if not isinstance(data, dict):
        raise SensorValidationError("sensor catalog root must be an object")
    _require_exact_keys(
        data,
        {"schema_version", "catalog_status", "claim_boundary", "sources", "sensors"},
        "catalog",
    )
    schema_version = _nonempty_string(data["schema_version"], "schema_version")
    if schema_version != "1.0.0":
        raise SensorValidationError(f"unsupported catalog schema: {schema_version}")
    status = _nonempty_string(data["catalog_status"], "catalog_status")
    if status != SYNTHETIC_STATUS:
        raise SensorValidationError(
            f"catalog_status must be {SYNTHETIC_STATUS!r}"
        )
    claim_boundary = _nonempty_string(data["claim_boundary"], "claim_boundary")

    if not isinstance(data["sources"], list) or not data["sources"]:
        raise SensorValidationError("sources must be a non-empty array")
    source_ids: set[str] = set()
    sources: list[dict[str, str]] = []
    for index, source in enumerate(data["sources"]):
        context = f"sources[{index}]"
        if not isinstance(source, dict):
            raise SensorValidationError(f"{context} must be an object")
        _require_exact_keys(source, {"id", "title", "url", "status", "use"}, context)
        parsed = {
            key: _nonempty_string(source[key], f"{context}.{key}")
            for key in ("id", "title", "url", "status", "use")
        }
        if parsed["id"] in source_ids:
            raise SensorValidationError(f"duplicate source id: {parsed['id']}")
        if parsed["status"] != SYNTHETIC_STATUS:
            raise SensorValidationError(
                f"{context}.status must be {SYNTHETIC_STATUS!r}"
            )
        if not parsed["url"].startswith(("https://", "http://")):
            raise SensorValidationError(f"{context}.url must be HTTP(S)")
        source_ids.add(parsed["id"])
        sources.append(parsed)

    if not isinstance(data["sensors"], list) or not data["sensors"]:
        raise SensorValidationError("sensors must be a non-empty array")
    sensors: list[SensorSpec] = []
    tags: set[str] = set()
    for index, sensor in enumerate(data["sensors"]):
        if not isinstance(sensor, dict):
            raise SensorValidationError(f"sensors[{index}] must be an object")
        parsed = _parse_sensor(sensor, index, source_ids)
        if parsed.tag in tags:
            raise SensorValidationError(f"duplicate sensor tag: {parsed.tag}")
        tags.add(parsed.tag)
        sensors.append(parsed)

    return SensorCatalog(
        schema_version=schema_version,
        catalog_status=status,
        claim_boundary=claim_boundary,
        sources=tuple(sources),
        sensors=tuple(sensors),
    )


@dataclass(frozen=True, slots=True)
class SensorReading:
    tag: str
    sequence: int
    elapsed_s: float
    true_value: float
    filtered_value: float
    analogue_value: float
    noisy_value: float
    quantized_value: float
    value: float | None
    quality: str
    dropout: bool
    held_last: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "sequence": self.sequence,
            "elapsed_s": self.elapsed_s,
            "true_value": self.true_value,
            "filtered_value": self.filtered_value,
            "analogue_value": self.analogue_value,
            "noisy_value": self.noisy_value,
            "quantized_value": self.quantized_value,
            "value": self.value,
            "quality": self.quality,
            "dropout": self.dropout,
            "held_last": self.held_last,
        }


def _quantize(value: float, increment: float) -> float:
    scaled = value / increment
    units = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    quantized = units * increment
    if not math.isfinite(quantized):
        raise SensorValidationError("quantization produced a non-finite value")
    return quantized


def _jsonify_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify_tuple(item) for item in value]
    if isinstance(value, list):
        return [_jsonify_tuple(item) for item in value]
    return value


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value


def _validate_finite_tree(value: Any, context: str = "state") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SensorValidationError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SensorValidationError(f"{context} has a non-string key")
            _validate_finite_tree(item, f"{context}.{key}")
        return
    raise SensorValidationError(f"{context} is not strict JSON data")


class SensorChain:
    """Stateful chain: lag -> affine/drift/hysteresis -> noise -> quantization -> dropout."""

    def __init__(
        self,
        specification: SensorSpec,
        *,
        seed: int,
        initial_true_value: float | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SensorValidationError("seed must be an integer")
        self.specification = specification
        self._rng = random.Random(seed)
        self._elapsed_s = 0.0
        self._sequence = 0
        self._filtered: float | None = None
        self._previous_true: float | None = None
        self._last_output: float | None = None
        if initial_true_value is not None:
            initial = _finite_number(initial_true_value, "initial_true_value")
            self._filtered = initial
            self._previous_true = initial

    def step(
        self,
        true_value: float,
        dt_s: float,
        *,
        force_dropout: bool | None = None,
    ) -> SensorReading:
        truth = _finite_number(true_value, "true_value")
        dt = _positive(dt_s, "dt_s")
        if force_dropout is not None and not isinstance(force_dropout, bool):
            raise SensorValidationError("force_dropout must be bool or None")

        if self._filtered is None:
            filtered = truth
        else:
            decay = math.exp(-dt / self.specification.lag_tau_s)
            filtered = truth + (self._filtered - truth) * decay

        direction = 0.0
        if self._previous_true is not None:
            if truth > self._previous_true:
                direction = 1.0
            elif truth < self._previous_true:
                direction = -1.0

        elapsed = self._elapsed_s + dt
        analogue = (
            self.specification.gain * filtered
            + self.specification.offset
            + self.specification.drift_per_hour * (elapsed / 3600.0)
            + self.specification.hysteresis * direction
        )
        noisy = analogue + self._rng.gauss(0.0, self.specification.noise_std)
        quantized = _quantize(noisy, self.specification.quantization)

        sampled_dropout = self._rng.random() < self.specification.dropout_probability
        dropout = sampled_dropout if force_dropout is None else force_dropout
        held_last = bool(
            dropout and self.specification.hold_last and self._last_output is not None
        )
        if dropout:
            output = self._last_output if held_last else None
            quality = "DROPOUT_HELD" if held_last else "DROPOUT"
        else:
            output = quantized
            quality = (
                "GOOD"
                if self.specification.minimum <= output <= self.specification.maximum
                else "OUT_OF_RANGE"
            )
            self._last_output = output

        self._filtered = filtered
        self._previous_true = truth
        self._elapsed_s = elapsed
        self._sequence += 1
        return SensorReading(
            tag=self.specification.tag,
            sequence=self._sequence,
            elapsed_s=elapsed,
            true_value=truth,
            filtered_value=filtered,
            analogue_value=analogue,
            noisy_value=noisy,
            quantized_value=quantized,
            value=output,
            quality=quality,
            dropout=dropout,
            held_last=held_last,
        )

    def export_state(self) -> dict[str, Any]:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "tag": self.specification.tag,
            "specification_sha256": self.specification.fingerprint,
            "elapsed_s": self._elapsed_s,
            "sequence": self._sequence,
            "filtered_value": self._filtered,
            "previous_true_value": self._previous_true,
            "last_output_value": self._last_output,
            "rng_state": _jsonify_tuple(self._rng.getstate()),
        }
        _validate_finite_tree(state)
        _canonical_json(state)
        return state

    def export_state_json(self) -> str:
        return _canonical_json(self.export_state())

    @classmethod
    def from_state(
        cls, specification: SensorSpec, state: Mapping[str, Any]
    ) -> "SensorChain":
        if not isinstance(state, dict):
            raise SensorValidationError("sensor state must be an object")
        _require_exact_keys(
            state,
            {
                "schema_version",
                "tag",
                "specification_sha256",
                "elapsed_s",
                "sequence",
                "filtered_value",
                "previous_true_value",
                "last_output_value",
                "rng_state",
            },
            "sensor state",
        )
        _validate_finite_tree(state)
        if state["schema_version"] != STATE_SCHEMA_VERSION:
            raise SensorValidationError("unsupported sensor state schema")
        if state["tag"] != specification.tag:
            raise SensorValidationError("sensor state tag does not match specification")
        if state["specification_sha256"] != specification.fingerprint:
            raise SensorValidationError("sensor state specification fingerprint mismatch")
        elapsed = _nonnegative(state["elapsed_s"], "state.elapsed_s")
        sequence = state["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise SensorValidationError("state.sequence must be a non-negative integer")

        optional_values: dict[str, float | None] = {}
        for key in ("filtered_value", "previous_true_value", "last_output_value"):
            item = state[key]
            optional_values[key] = (
                None if item is None else _finite_number(item, f"state.{key}")
            )
        if (optional_values["filtered_value"] is None) != (
            optional_values["previous_true_value"] is None
        ):
            raise SensorValidationError(
                "filtered and previous-true state must both be set or both be null"
            )

        restored = cls(specification, seed=0)
        try:
            restored._rng.setstate(_tupleize(state["rng_state"]))
        except (TypeError, ValueError) as exc:
            raise SensorValidationError("invalid random-generator state") from exc
        restored._elapsed_s = elapsed
        restored._sequence = sequence
        restored._filtered = optional_values["filtered_value"]
        restored._previous_true = optional_values["previous_true_value"]
        restored._last_output = optional_values["last_output_value"]
        return restored

    @classmethod
    def from_state_json(
        cls, specification: SensorSpec, text: str
    ) -> "SensorChain":
        state = strict_json_loads(text)
        if not isinstance(state, dict):
            raise SensorValidationError("sensor state JSON root must be an object")
        return cls.from_state(specification, state)


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    reference: float
    indicated: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference", _finite_number(self.reference, "point.reference")
        )
        object.__setattr__(
            self, "indicated", _finite_number(self.indicated, "point.indicated")
        )

    def to_dict(self) -> dict[str, float]:
        return {"reference": self.reference, "indicated": self.indicated}


def _normalise_points(
    points: Sequence[CalibrationPoint | Mapping[str, Any]], context: str
) -> tuple[CalibrationPoint, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise SensorValidationError(f"{context} must be a sequence")
    normalised: list[CalibrationPoint] = []
    seen: set[tuple[float, float]] = set()
    for index, point in enumerate(points):
        if isinstance(point, CalibrationPoint):
            parsed = point
        elif isinstance(point, Mapping):
            _require_exact_keys(
                point, {"reference", "indicated"}, f"{context}[{index}]"
            )
            parsed = CalibrationPoint(point["reference"], point["indicated"])
        else:
            raise SensorValidationError(f"{context}[{index}] is not a calibration point")
        pair = (parsed.reference, parsed.indicated)
        if pair in seen:
            raise SensorValidationError(f"duplicate calibration point in {context}: {pair}")
        seen.add(pair)
        normalised.append(parsed)
    if len(normalised) < 3:
        raise SensorValidationError(f"{context} requires at least three points")
    return tuple(normalised)


@dataclass(frozen=True, slots=True)
class OLSFit:
    intercept: float
    slope: float
    residual_std_indicated: float
    reference_minimum: float
    reference_maximum: float
    point_count: int
    variance_intercept: float
    covariance_intercept_slope: float
    variance_slope: float

    def inverse(self, indicated: float) -> float:
        reading = _finite_number(indicated, "indicated")
        corrected = (reading - self.intercept) / self.slope
        if not math.isfinite(corrected):
            raise SensorValidationError("inverse calibration produced non-finite value")
        return corrected

    def parameter_standard_uncertainty(self, indicated: float) -> float:
        reading = _finite_number(indicated, "indicated")
        corrected = self.inverse(reading)
        sensitivity_intercept = -1.0 / self.slope
        sensitivity_slope = -corrected / self.slope
        variance = (
            sensitivity_intercept**2 * self.variance_intercept
            + sensitivity_slope**2 * self.variance_slope
            + 2.0
            * sensitivity_intercept
            * sensitivity_slope
            * self.covariance_intercept_slope
        )
        tolerance = 1e-12 * max(
            1.0,
            abs(self.variance_intercept),
            abs(self.variance_slope),
        )
        if variance < -tolerance:
            raise SensorValidationError("fit covariance produced negative uncertainty")
        return math.sqrt(max(0.0, variance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "residual_std_indicated": self.residual_std_indicated,
            "reference_minimum": self.reference_minimum,
            "reference_maximum": self.reference_maximum,
            "point_count": self.point_count,
            "variance_intercept": self.variance_intercept,
            "covariance_intercept_slope": self.covariance_intercept_slope,
            "variance_slope": self.variance_slope,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], context: str) -> "OLSFit":
        keys = {
            "intercept",
            "slope",
            "residual_std_indicated",
            "reference_minimum",
            "reference_maximum",
            "point_count",
            "variance_intercept",
            "covariance_intercept_slope",
            "variance_slope",
        }
        _require_exact_keys(value, keys, context)
        point_count = value["point_count"]
        if isinstance(point_count, bool) or not isinstance(point_count, int):
            raise SensorValidationError(f"{context}.point_count must be integer")
        fit = cls(
            intercept=_finite_number(value["intercept"], f"{context}.intercept"),
            slope=_positive(value["slope"], f"{context}.slope"),
            residual_std_indicated=_nonnegative(
                value["residual_std_indicated"],
                f"{context}.residual_std_indicated",
            ),
            reference_minimum=_finite_number(
                value["reference_minimum"], f"{context}.reference_minimum"
            ),
            reference_maximum=_finite_number(
                value["reference_maximum"], f"{context}.reference_maximum"
            ),
            point_count=point_count,
            variance_intercept=_nonnegative(
                value["variance_intercept"], f"{context}.variance_intercept"
            ),
            covariance_intercept_slope=_finite_number(
                value["covariance_intercept_slope"],
                f"{context}.covariance_intercept_slope",
            ),
            variance_slope=_nonnegative(
                value["variance_slope"], f"{context}.variance_slope"
            ),
        )
        if fit.point_count < 3 or fit.reference_minimum >= fit.reference_maximum:
            raise SensorValidationError(f"{context} contains invalid fit bounds")
        return fit


def fit_ols(
    points: Sequence[CalibrationPoint | Mapping[str, Any]],
    *,
    context: str = "calibration points",
) -> OLSFit:
    """Fit indicated ``y = a + b * reference x`` by ordinary least squares."""
    parsed = _normalise_points(points, context)
    references = [point.reference for point in parsed]
    indications = [point.indicated for point in parsed]
    count = len(parsed)
    mean_reference = math.fsum(references) / count
    mean_indicated = math.fsum(indications) / count
    sxx = math.fsum((value - mean_reference) ** 2 for value in references)
    if sxx <= 0.0:
        raise SensorValidationError(f"{context} must contain distinct references")
    sxy = math.fsum(
        (point.reference - mean_reference) * (point.indicated - mean_indicated)
        for point in parsed
    )
    slope = sxy / sxx
    if not math.isfinite(slope) or slope <= 0.0:
        raise SensorValidationError(f"{context} fitted slope must be positive")
    intercept = mean_indicated - slope * mean_reference
    residuals = [
        point.indicated - (intercept + slope * point.reference) for point in parsed
    ]
    sse = math.fsum(residual * residual for residual in residuals)
    residual_variance = max(0.0, sse / (count - 2))
    residual_std = math.sqrt(residual_variance)
    variance_slope = residual_variance / sxx
    variance_intercept = residual_variance * (
        1.0 / count + mean_reference**2 / sxx
    )
    covariance = -mean_reference * residual_variance / sxx
    values = (
        intercept,
        slope,
        residual_std,
        variance_slope,
        variance_intercept,
        covariance,
    )
    if not all(math.isfinite(value) for value in values):
        raise SensorValidationError(f"{context} OLS result is non-finite")
    return OLSFit(
        intercept=intercept,
        slope=slope,
        residual_std_indicated=residual_std,
        reference_minimum=min(references),
        reference_maximum=max(references),
        point_count=count,
        variance_intercept=variance_intercept,
        covariance_intercept_slope=covariance,
        variance_slope=variance_slope,
    )


def combined_standard_uncertainty(components: Mapping[str, float]) -> float:
    """Combine independent standard uncertainties by NIST TN 1297 RSS."""
    if not isinstance(components, Mapping) or not components:
        raise SensorValidationError("uncertainty components must be a non-empty mapping")
    squares: list[float] = []
    for name, value in components.items():
        _nonempty_string(name, "uncertainty component name")
        standard_uncertainty = _nonnegative(value, f"uncertainty.{name}")
        square = standard_uncertainty * standard_uncertainty
        if not math.isfinite(square):
            raise SensorValidationError(f"uncertainty.{name} is too large")
        squares.append(square)
    combined = math.sqrt(math.fsum(squares))
    if not math.isfinite(combined):
        raise SensorValidationError("combined uncertainty is non-finite")
    return combined


def _normalise_uncertainty_components(
    components: Mapping[str, float], context: str
) -> tuple[tuple[str, float], ...]:
    """Validate, normalize, and deterministically order an uncertainty budget."""
    if not isinstance(components, Mapping) or not components:
        raise SensorValidationError(f"{context} must be a non-empty mapping")
    normalised: list[tuple[str, float]] = []
    for name, value in components.items():
        clean_name = _nonempty_string(name, f"{context} component name")
        normalised.append(
            (clean_name, _nonnegative(value, f"{context}.{clean_name}"))
        )
    normalised.sort(key=lambda item: item[0])
    combined_standard_uncertainty(dict(normalised))
    return tuple(normalised)


def expanded_uncertainty(combined: float, coverage_factor: float) -> float:
    uc = _nonnegative(combined, "combined standard uncertainty")
    factor = _positive(coverage_factor, "coverage factor")
    expanded = factor * uc
    if not math.isfinite(expanded):
        raise SensorValidationError("expanded uncertainty is non-finite")
    return expanded


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    record_id: str
    sensor_tag: str
    status: str
    calibrated_at: str
    due_at: str
    as_found_points: tuple[CalibrationPoint, ...]
    as_left_points: tuple[CalibrationPoint, ...]
    as_found_fit: OLSFit
    as_left_fit: OLSFit
    uncertainty_components: tuple[tuple[str, float], ...]
    coverage_factor: float
    claim_boundary: str

    def status_at(self, at: datetime | str) -> str:
        instant = _utc_datetime(at, "status time")
        calibrated = _utc_datetime(self.calibrated_at, "calibrated_at")
        due = _utc_datetime(self.due_at, "due_at")
        if instant < calibrated:
            return "NOT_YET_VALID"
        if instant >= due:
            return "EXPIRED"
        return "VALID"

    def is_due(self, at: datetime | str) -> bool:
        return self.status_at(at) == "EXPIRED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "record_id": self.record_id,
            "sensor_tag": self.sensor_tag,
            "status": self.status,
            "calibrated_at": self.calibrated_at,
            "due_at": self.due_at,
            "as_found_points": [point.to_dict() for point in self.as_found_points],
            "as_left_points": [point.to_dict() for point in self.as_left_points],
            "as_found_fit": self.as_found_fit.to_dict(),
            "as_left_fit": self.as_left_fit.to_dict(),
            "uncertainty_components": dict(self.uncertainty_components),
            "coverage_factor": self.coverage_factor,
            "claim_boundary": self.claim_boundary,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationRecord":
        if not isinstance(value, dict):
            raise SensorValidationError("calibration record must be an object")
        _require_exact_keys(
            value,
            {
                "schema_version",
                "record_id",
                "sensor_tag",
                "status",
                "calibrated_at",
                "due_at",
                "as_found_points",
                "as_left_points",
                "as_found_fit",
                "as_left_fit",
                "uncertainty_components",
                "coverage_factor",
                "claim_boundary",
            },
            "calibration record",
        )
        if value["schema_version"] != CALIBRATION_SCHEMA_VERSION:
            raise SensorValidationError("unsupported calibration record schema")
        status = _nonempty_string(value["status"], "record.status")
        if status != SYNTHETIC_STATUS:
            raise SensorValidationError(
                f"record.status must be {SYNTHETIC_STATUS!r}"
            )
        calibrated = _utc_datetime(value["calibrated_at"], "record.calibrated_at")
        due = _utc_datetime(value["due_at"], "record.due_at")
        if due <= calibrated:
            raise SensorValidationError("record.due_at must follow calibrated_at")
        found = _normalise_points(value["as_found_points"], "record.as_found_points")
        left = _normalise_points(value["as_left_points"], "record.as_left_points")
        as_found_fit = OLSFit.from_mapping(value["as_found_fit"], "as_found_fit")
        as_left_fit = OLSFit.from_mapping(value["as_left_fit"], "as_left_fit")
        # Re-fitting detects corrupted or inconsistent exported point/fit pairs.
        if _canonical_json(fit_ols(found).to_dict()) != _canonical_json(
            as_found_fit.to_dict()
        ):
            raise SensorValidationError("as-found fit does not match exported points")
        if _canonical_json(fit_ols(left).to_dict()) != _canonical_json(
            as_left_fit.to_dict()
        ):
            raise SensorValidationError("as-left fit does not match exported points")
        component_data = value["uncertainty_components"]
        if not isinstance(component_data, dict) or not component_data:
            raise SensorValidationError("record uncertainty components are missing")
        components = _normalise_uncertainty_components(
            component_data,
            "record.uncertainty",
        )
        return cls(
            record_id=_nonempty_string(value["record_id"], "record.record_id"),
            sensor_tag=_nonempty_string(value["sensor_tag"], "record.sensor_tag"),
            status=status,
            calibrated_at=_iso_utc(calibrated),
            due_at=_iso_utc(due),
            as_found_points=found,
            as_left_points=left,
            as_found_fit=as_found_fit,
            as_left_fit=as_left_fit,
            uncertainty_components=components,
            coverage_factor=_positive(
                value["coverage_factor"], "record.coverage_factor"
            ),
            claim_boundary=_nonempty_string(
                value["claim_boundary"], "record.claim_boundary"
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "CalibrationRecord":
        value = strict_json_loads(text)
        if not isinstance(value, dict):
            raise SensorValidationError("calibration record JSON root must be object")
        return cls.from_dict(value)


def create_calibration_record(
    specification: SensorSpec,
    *,
    record_id: str,
    calibrated_at: datetime | str,
    as_found_points: Sequence[CalibrationPoint | Mapping[str, Any]],
    as_left_points: Sequence[CalibrationPoint | Mapping[str, Any]],
    uncertainty_components: Mapping[str, float] | None = None,
    coverage_factor: float | None = None,
) -> CalibrationRecord:
    """Create a synthetic as-found/as-left record and its OLS fits."""
    clean_id = _nonempty_string(record_id, "record_id")
    instant = _utc_datetime(calibrated_at, "calibrated_at")
    found = _normalise_points(as_found_points, "as_found_points")
    left = _normalise_points(as_left_points, "as_left_points")
    components = (
        specification.uncertainty_component_map
        if uncertainty_components is None
        else dict(uncertainty_components)
    )
    normalised_components = _normalise_uncertainty_components(
        components,
        "uncertainty",
    )
    factor = (
        specification.coverage_factor
        if coverage_factor is None
        else _positive(coverage_factor, "coverage_factor")
    )
    due = instant + timedelta(days=specification.calibration_interval_days)
    return CalibrationRecord(
        record_id=clean_id,
        sensor_tag=specification.tag,
        status=SYNTHETIC_STATUS,
        calibrated_at=_iso_utc(instant),
        due_at=_iso_utc(due),
        as_found_points=found,
        as_left_points=left,
        as_found_fit=fit_ols(found, context="as_found_points"),
        as_left_fit=fit_ols(left, context="as_left_points"),
        uncertainty_components=normalised_components,
        coverage_factor=factor,
        claim_boundary=(
            "Synthetic reference record only; no unbroken calibration chain, "
            "metrological traceability, accreditation, or fitness for use is claimed."
        ),
    )


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    status: str
    reason: str
    sensor_tag: str
    record_id: str
    indicated: float
    corrected: float | None
    combined_standard_uncertainty: float | None
    coverage_factor: float | None
    expanded_uncertainty: float | None
    lower_bound: float | None
    upper_bound: float | None
    in_calibration_range: bool | None
    uncertainty_components: tuple[tuple[str, float], ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "sensor_tag": self.sensor_tag,
            "record_id": self.record_id,
            "indicated": self.indicated,
            "corrected": self.corrected,
            "combined_standard_uncertainty": self.combined_standard_uncertainty,
            "coverage_factor": self.coverage_factor,
            "expanded_uncertainty": self.expanded_uncertainty,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "in_calibration_range": self.in_calibration_range,
            "uncertainty_components": dict(self.uncertainty_components),
            "claim_boundary": self.claim_boundary,
        }


def _unavailable_correction(
    specification: SensorSpec,
    record: CalibrationRecord,
    indicated: float,
    status: str,
    reason: str,
    *,
    corrected: float | None = None,
    in_range: bool | None = None,
) -> CorrectionResult:
    return CorrectionResult(
        status=status,
        reason=reason,
        sensor_tag=specification.tag,
        record_id=record.record_id,
        indicated=indicated,
        corrected=corrected,
        combined_standard_uncertainty=None,
        coverage_factor=None,
        expanded_uncertainty=None,
        lower_bound=None,
        upper_bound=None,
        in_calibration_range=in_range,
        uncertainty_components=(),
        claim_boundary=record.claim_boundary,
    )


def correct_with_uncertainty(
    specification: SensorSpec,
    record: CalibrationRecord,
    indicated: float,
    *,
    observed_at: datetime | str,
    additional_uncertainty_components: Mapping[str, float] | None = None,
    allow_extrapolation: bool = False,
) -> CorrectionResult:
    """Apply the as-left inverse correction and a GUM-style uncertainty budget."""
    reading = _finite_number(indicated, "indicated")
    if not isinstance(allow_extrapolation, bool):
        raise SensorValidationError("allow_extrapolation must be boolean")
    if record.sensor_tag != specification.tag:
        raise SensorValidationError("calibration record belongs to a different sensor")
    temporal_status = record.status_at(observed_at)
    if temporal_status != "VALID":
        return _unavailable_correction(
            specification,
            record,
            reading,
            temporal_status,
            "calibration record is not valid at the observation time",
        )
    if not specification.minimum <= reading <= specification.maximum:
        return _unavailable_correction(
            specification,
            record,
            reading,
            "OUT_OF_RANGE",
            "indicated value is outside the engineering range",
            in_range=False,
        )

    corrected = record.as_left_fit.inverse(reading)
    if not specification.minimum <= corrected <= specification.maximum:
        return _unavailable_correction(
            specification,
            record,
            reading,
            "OUT_OF_RANGE",
            "inverse-corrected value is outside the engineering range",
            corrected=corrected,
            in_range=False,
        )
    calibration_range = (
        record.as_left_fit.reference_minimum <= corrected
        <= record.as_left_fit.reference_maximum
    )
    if not calibration_range and not allow_extrapolation:
        return _unavailable_correction(
            specification,
            record,
            reading,
            "OUT_OF_RANGE",
            "inverse-corrected value is outside the calibrated reference range",
            corrected=corrected,
            in_range=False,
        )

    components = dict(record.uncertainty_components)
    fit_parameter = record.as_left_fit.parameter_standard_uncertainty(reading)
    fit_residual = record.as_left_fit.residual_std_indicated / abs(
        record.as_left_fit.slope
    )
    for name, value in (
        ("ols_parameter_covariance", fit_parameter),
        ("ols_residual", fit_residual),
    ):
        if name in components:
            raise SensorValidationError(f"reserved uncertainty component: {name}")
        components[name] = value
    if additional_uncertainty_components is not None:
        if not isinstance(additional_uncertainty_components, Mapping):
            raise SensorValidationError(
                "additional_uncertainty_components must be a mapping"
            )
        for name, value in additional_uncertainty_components.items():
            clean_name = _nonempty_string(name, "additional uncertainty name")
            if clean_name in components:
                raise SensorValidationError(
                    f"duplicate uncertainty component: {clean_name}"
                )
            components[clean_name] = _nonnegative(
                value, f"additional uncertainty.{clean_name}"
            )

    combined = combined_standard_uncertainty(components)
    expanded = expanded_uncertainty(combined, record.coverage_factor)
    lower = corrected - expanded
    upper = corrected + expanded
    if not all(math.isfinite(value) for value in (lower, upper)):
        raise SensorValidationError("correction interval is non-finite")
    return CorrectionResult(
        status="VALID",
        reason=(
            "inverse OLS correction with synthetic uncertainty budget"
            if calibration_range
            else "inverse OLS correction extrapolated by explicit request"
        ),
        sensor_tag=specification.tag,
        record_id=record.record_id,
        indicated=reading,
        corrected=corrected,
        combined_standard_uncertainty=combined,
        coverage_factor=record.coverage_factor,
        expanded_uncertainty=expanded,
        lower_bound=lower,
        upper_bound=upper,
        in_calibration_range=calibration_range,
        uncertainty_components=tuple(sorted(components.items())),
        claim_boundary=record.claim_boundary,
    )


@dataclass(frozen=True, slots=True)
class GuardBandDecision:
    status: str
    disposition: str
    reason: str
    sensor_tag: str
    direction: str
    critical_limit: float
    guarded_value: float | None
    correction: CorrectionResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "disposition": self.disposition,
            "reason": self.reason,
            "sensor_tag": self.sensor_tag,
            "direction": self.direction,
            "critical_limit": self.critical_limit,
            "guarded_value": self.guarded_value,
            "correction": None if self.correction is None else self.correction.to_dict(),
        }


def evaluate_guard_band(
    specification: SensorSpec,
    correction: CorrectionResult,
    *,
    critical_limit: float | None = None,
    direction: str | None = None,
) -> GuardBandDecision:
    """Evaluate a lower/upper guard band; uncertainty overlap fails closed UNKNOWN."""
    limit = (
        specification.critical_limit
        if critical_limit is None
        else _finite_number(critical_limit, "critical_limit")
    )
    selected_direction = specification.guard_direction if direction is None else direction
    if selected_direction not in {"lower", "upper"}:
        raise SensorValidationError("guard direction must be 'lower' or 'upper'")
    if correction.sensor_tag != specification.tag:
        raise SensorValidationError("correction belongs to a different sensor")
    if correction.status != "VALID":
        return GuardBandDecision(
            status="UNKNOWN",
            disposition="HOLD",
            reason=f"correction unavailable: {correction.status}",
            sensor_tag=specification.tag,
            direction=selected_direction,
            critical_limit=limit,
            guarded_value=None,
            correction=correction,
        )
    interval = (
        correction.corrected,
        correction.lower_bound,
        correction.upper_bound,
        correction.expanded_uncertainty,
    )
    if any(value is None for value in interval) or not all(
        math.isfinite(float(value)) for value in interval if value is not None
    ):
        return GuardBandDecision(
            status="UNKNOWN",
            disposition="HOLD",
            reason="valid-labelled correction has an incomplete or non-finite interval",
            sensor_tag=specification.tag,
            direction=selected_direction,
            critical_limit=limit,
            guarded_value=None,
            correction=correction,
        )
    corrected = float(correction.corrected)
    lower_bound = float(correction.lower_bound)
    upper_bound = float(correction.upper_bound)
    expanded = float(correction.expanded_uncertainty)
    if (
        expanded < 0.0
        or lower_bound > corrected
        or corrected > upper_bound
        or lower_bound > upper_bound
    ):
        return GuardBandDecision(
            status="UNKNOWN",
            disposition="HOLD",
            reason="valid-labelled correction has an inconsistent interval",
            sensor_tag=specification.tag,
            direction=selected_direction,
            critical_limit=limit,
            guarded_value=None,
            correction=correction,
        )
    if selected_direction == "lower":
        guarded = lower_bound
        if guarded >= limit:
            status, disposition, reason = (
                "PASS",
                "NO_AUTOMATIC_RELEASE",
                "expanded-uncertainty lower bound meets the minimum",
            )
        elif upper_bound < limit:
            status, disposition, reason = (
                "FAIL",
                "HOLD",
                "expanded-uncertainty interval is below the minimum",
            )
        else:
            status, disposition, reason = (
                "UNKNOWN",
                "HOLD",
                "expanded-uncertainty interval overlaps the minimum",
            )
    else:
        guarded = upper_bound
        if guarded <= limit:
            status, disposition, reason = (
                "PASS",
                "NO_AUTOMATIC_RELEASE",
                "expanded-uncertainty upper bound meets the maximum",
            )
        elif lower_bound > limit:
            status, disposition, reason = (
                "FAIL",
                "HOLD",
                "expanded-uncertainty interval is above the maximum",
            )
        else:
            status, disposition, reason = (
                "UNKNOWN",
                "HOLD",
                "expanded-uncertainty interval overlaps the maximum",
            )
    return GuardBandDecision(
        status=status,
        disposition=disposition,
        reason=reason,
        sensor_tag=specification.tag,
        direction=selected_direction,
        critical_limit=limit,
        guarded_value=guarded,
        correction=correction,
    )


def evaluate_guarded_reading(
    specification: SensorSpec,
    reading: SensorReading,
    record: CalibrationRecord | None,
    *,
    observed_at: datetime | str,
    additional_uncertainty_components: Mapping[str, float] | None = None,
) -> GuardBandDecision:
    """Fail-closed convenience API for dynamic readings and calibration records."""
    if reading.tag != specification.tag:
        raise SensorValidationError("reading belongs to a different sensor")
    if record is None:
        return GuardBandDecision(
            status="UNKNOWN",
            disposition="HOLD",
            reason="missing calibration record",
            sensor_tag=specification.tag,
            direction=specification.guard_direction,
            critical_limit=specification.critical_limit,
            guarded_value=None,
            correction=None,
        )
    if reading.quality != "GOOD" or reading.value is None:
        return GuardBandDecision(
            status="UNKNOWN",
            disposition="HOLD",
            reason=f"sensor quality is {reading.quality}",
            sensor_tag=specification.tag,
            direction=specification.guard_direction,
            critical_limit=specification.critical_limit,
            guarded_value=None,
            correction=None,
        )
    correction = correct_with_uncertainty(
        specification,
        record,
        reading.value,
        observed_at=observed_at,
        additional_uncertainty_components=additional_uncertainty_components,
    )
    return evaluate_guard_band(specification, correction)


__all__ = [
    "CATALOG_PATH",
    "SYNTHETIC_STATUS",
    "CalibrationPoint",
    "CalibrationRecord",
    "CorrectionResult",
    "GuardBandDecision",
    "OLSFit",
    "SensorCatalog",
    "SensorChain",
    "SensorReading",
    "SensorSpec",
    "SensorValidationError",
    "combined_standard_uncertainty",
    "correct_with_uncertainty",
    "create_calibration_record",
    "evaluate_guard_band",
    "evaluate_guarded_reading",
    "expanded_uncertainty",
    "fit_ols",
    "load_sensor_catalog",
    "strict_json_loads",
]
