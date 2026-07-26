#!/usr/bin/env python3
"""Vendor-neutral PLC shadow for reference HTST rows.

``ReferencePLC`` consumes the observable/status fields already emitted by an
HTST scenario or campaign and evaluates a separate cause-and-effect/state
machine.  It never writes back to :class:`model.HTSTSimulator`, never drives an
actuator, and is not a deployable safety PLC.  Its purpose is deterministic
research traceability between a reference P&ID, an explicit I/O image, Python
logic and the companion IEC 61131-3 Structured Text listing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Package and direct-script/test imports are both supported.
    from .reference_plant import load_reference_plant
except ImportError:  # pragma: no cover - direct import used by repository tests
    from reference_plant import load_reference_plant


HERE = Path(__file__).resolve().parent
DEFAULT_CAUSE_EFFECT_PATH = HERE / "plc" / "cause_effect.json"
DEFAULT_ST_PATH = HERE / "plc" / "htst_reference.st"
CAUSE_EFFECT_SCHEMA_VERSION = "1.0.0"


class PLCLogicError(ValueError):
    """Raised for an invalid I/O image or reference PLC contract."""


class PLCState(str, Enum):
    OFF = "OFF"
    STARTUP = "STARTUP"
    RECIRCULATE = "RECIRCULATE"
    FORWARD = "FORWARD"
    CIP = "CIP"
    TRIP = "TRIP"


class CIPRecipeState(str, Enum):
    IDLE = "IDLE"
    PRE_RINSE = "PRE_RINSE"
    CAUSTIC = "CAUSTIC"
    INTERMEDIATE_RINSE = "INTERMEDIATE_RINSE"
    ACID = "ACID"
    FINAL_RINSE = "FINAL_RINSE"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class PLCSettings:
    scan_time_s: float = 0.5
    startup_delay_s: float = 1.0
    forward_confirmation_s: float = 1.0
    diversion_threshold_c: float = 72.0
    forward_temperature_margin_c: float = 0.30
    minimum_holding_time_s: float = 15.0
    required_differential_pressure_bar: float = 0.50
    sensor_disagreement_limit_c: float = 0.75
    leak_alarm_fraction: float = 0.005
    cip_pre_rinse_s: float = 60.0
    cip_caustic_s: float = 180.0
    cip_intermediate_rinse_s: float = 60.0
    cip_acid_s: float = 120.0
    cip_final_rinse_s: float = 120.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PLCLogicError(f"{item.name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise PLCLogicError(f"{item.name} must be finite and non-negative")
        for name in (
            "scan_time_s",
            "startup_delay_s",
            "forward_confirmation_s",
            "minimum_holding_time_s",
            "required_differential_pressure_bar",
            "cip_pre_rinse_s",
            "cip_caustic_s",
            "cip_intermediate_rinse_s",
            "cip_acid_s",
            "cip_final_rinse_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise PLCLogicError(f"{name} must be positive")


@dataclass(frozen=True)
class PLCInputs:
    """One explicit, immutable PLC input image derived from a simulator row."""

    time_s: float
    step_dt_s: float
    operator_start: bool
    operator_stop: bool
    trip_ack: bool
    trip_reset: bool
    emergency_stop: bool
    power_good: bool
    temperature_sensor_quality_ok: bool
    cip_active: bool
    cip_release_permissive: bool
    safety_temperature_c: float
    estimated_fastest_residence_time_s: float
    measured_flow_l_h: float
    maximum_safe_flow_l_h: float
    measured_differential_pressure_bar: float
    sensor_disagreement_c: float
    leak_detector_signal_fraction: float
    fdv_position_feedback: float
    product_interface_fraction: float
    fdv_mismatch: bool
    regenerator_leak: bool
    low_temperature: bool
    low_holding_time: bool
    high_flow: bool
    low_differential_pressure: bool
    sensor_disagreement: bool
    plant_mode: str


@dataclass(frozen=True)
class PLCOutputs:
    """One explicit PLC output image; these values are shadow commands only."""

    feed_pump_command: bool = False
    booster_pump_command: bool = False
    cip_pump_command: bool = False
    steam_enable: bool = False
    fdv_command_forward: bool = False
    fdv_command_divert: bool = True
    cip_rinse_valve: bool = False
    cip_alkali_valve: bool = False
    cip_acid_valve: bool = False
    cip_drain_valve: bool = False


@dataclass(frozen=True)
class CauseDefinition:
    cause_id: str
    input_name: str
    predicate: str
    classification: str
    latching: bool
    effect_state: str
    actions: tuple[str, ...]
    python_rule: str
    st_marker: str


SUPPORTED_CAUSE_RULES = frozenset(
    {
        "TRIP_ESTOP",
        "TRIP_POWER_BAD",
        "TRIP_SENSOR_BAD",
        "TRIP_FDV_MISMATCH",
        "TRIP_REGENERATOR_LEAK",
        "DIVERT_LOW_TEMPERATURE",
        "DIVERT_LOW_HOLDING_TIME",
        "DIVERT_HIGH_FLOW",
        "DIVERT_LOW_DIFFERENTIAL_PRESSURE",
        "DIVERT_SENSOR_DISAGREEMENT",
        "DIVERT_CIP_ACTIVE",
        "DIVERT_CIP_NOT_RELEASED",
    }
)


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PLCLogicError(f"{context} must be an object")
    return value


def _required_number(row: Mapping[str, object], name: str) -> float:
    if name not in row:
        raise PLCLogicError(f"PLC input row is missing {name!r}")
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PLCLogicError(f"PLC input {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PLCLogicError(f"PLC input {name!r} must be finite")
    return result


def _optional_number(row: Mapping[str, object], name: str, default: float) -> float:
    if name not in row:
        return default
    return _required_number(row, name)


def _bool_value(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
    raise PLCLogicError(f"PLC input {name!r} must be boolean or 0/1")


def _required_bool(row: Mapping[str, object], name: str) -> bool:
    if name not in row:
        raise PLCLogicError(f"PLC input row is missing {name!r}")
    return _bool_value(row[name], name)


def _optional_bool(row: Mapping[str, object], name: str, default: bool) -> bool:
    return default if name not in row else _bool_value(row[name], name)


def _load_cause_effect(path: str | Path = DEFAULT_CAUSE_EFFECT_PATH) -> tuple[str, tuple[CauseDefinition, ...]]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PLCLogicError(f"cannot read cause/effect matrix {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PLCLogicError(f"invalid cause/effect JSON {source}: {exc}") from exc
    root = _as_mapping(value, "cause/effect matrix")
    if root.get("schema_version") != CAUSE_EFFECT_SCHEMA_VERSION:
        raise PLCLogicError("unsupported cause/effect schema_version")
    if root.get("shadow_only") is not True:
        raise PLCLogicError("cause/effect matrix must declare shadow_only=true")
    status = root.get("status")
    if not isinstance(status, str) or "does not control" not in status.lower():
        raise PLCLogicError("cause/effect status must forbid physical control")
    raw_causes = root.get("causes")
    if not isinstance(raw_causes, list) or not raw_causes:
        raise PLCLogicError("cause/effect causes must be a non-empty array")
    input_names = {item.name for item in fields(PLCInputs)}
    known_states = {state.value for state in PLCState}
    definitions: list[CauseDefinition] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_causes):
        item = _as_mapping(raw, f"causes[{index}]")
        string_names = (
            "cause_id",
            "input_name",
            "predicate",
            "classification",
            "effect_state",
            "python_rule",
            "st_marker",
        )
        strings: dict[str, str] = {}
        for name in string_names:
            raw_value = item.get(name)
            if not isinstance(raw_value, str) or not raw_value:
                raise PLCLogicError(f"causes[{index}].{name} must be non-empty")
            strings[name] = raw_value
        if strings["cause_id"] in identifiers:
            raise PLCLogicError(f"duplicate cause_id {strings['cause_id']!r}")
        identifiers.add(strings["cause_id"])
        if strings["input_name"] not in input_names:
            raise PLCLogicError(
                f"cause {strings['cause_id']!r} references unknown PLC input"
            )
        if strings["predicate"] not in {"TRUE", "FALSE"}:
            raise PLCLogicError("cause predicate must be TRUE or FALSE")
        if strings["classification"] not in {"TRIP", "DIVERT"}:
            raise PLCLogicError("cause classification must be TRIP or DIVERT")
        latching = item.get("latching")
        if not isinstance(latching, bool):
            raise PLCLogicError("cause latching must be boolean")
        if latching != (strings["classification"] == "TRIP"):
            raise PLCLogicError("only TRIP causes may be latching in this contract")
        if strings["effect_state"] not in known_states:
            raise PLCLogicError("cause effect_state is not a PLC state")
        if strings["python_rule"] not in SUPPORTED_CAUSE_RULES:
            raise PLCLogicError("cause references an unsupported Python rule")
        actions = item.get("actions")
        if (
            not isinstance(actions, list)
            or not actions
            or not all(isinstance(action, str) and action for action in actions)
        ):
            raise PLCLogicError("cause actions must be a non-empty string array")
        definitions.append(
            CauseDefinition(
                cause_id=strings["cause_id"],
                input_name=strings["input_name"],
                predicate=strings["predicate"],
                classification=strings["classification"],
                latching=latching,
                effect_state=strings["effect_state"],
                actions=tuple(actions),
                python_rule=strings["python_rule"],
                st_marker=strings["st_marker"],
            )
        )
    rules = {definition.python_rule for definition in definitions}
    if rules != set(SUPPORTED_CAUSE_RULES):
        raise PLCLogicError(
            "cause/effect Python rules are incomplete: "
            f"missing={sorted(SUPPORTED_CAUSE_RULES - rules)}, "
            f"extra={sorted(rules - SUPPORTED_CAUSE_RULES)}"
        )
    controller_id = root.get("controller_id")
    if not isinstance(controller_id, str) or not controller_id:
        raise PLCLogicError("cause/effect controller_id must be non-empty")
    return controller_id, tuple(definitions)


def validate_plc_traceability(
    cause_effect_path: str | Path = DEFAULT_CAUSE_EFFECT_PATH,
    st_path: str | Path = DEFAULT_ST_PATH,
) -> dict[str, object]:
    """Verify one-to-one cause IDs and trace markers across JSON/Python/ST."""

    controller_id, definitions = _load_cause_effect(cause_effect_path)
    try:
        source = Path(st_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PLCLogicError(f"cannot read Structured Text reference {st_path}: {exc}") from exc
    missing = [definition.st_marker for definition in definitions if definition.st_marker not in source]
    duplicates = [
        definition.st_marker
        for definition in definitions
        if source.count(definition.st_marker) != 1
    ]
    if missing or duplicates:
        raise PLCLogicError(
            "Structured Text trace markers are incomplete or non-unique: "
            f"missing={missing}, non_unique={duplicates}"
        )
    for state in PLCState:
        if state.value not in source:
            raise PLCLogicError(f"Structured Text does not name state {state.value}")
    return {
        "valid": True,
        "controller_id": controller_id,
        "cause_count": len(definitions),
        "python_rule_count": len(SUPPORTED_CAUSE_RULES),
        "st_marker_count": len(definitions),
        "shadow_only": True,
    }


def _cause_active(inputs: PLCInputs, definition: CauseDefinition) -> bool:
    value = getattr(inputs, definition.input_name)
    if not isinstance(value, bool):
        raise PLCLogicError(
            f"cause input {definition.input_name!r} is not boolean in the I/O image"
        )
    return value if definition.predicate == "TRUE" else not value


class ReferencePLC:
    """Deterministic vendor-neutral HTST PLC shadow.

    The controller has no reference to a simulator object and returns a fresh
    dictionary for every scan.  Therefore callers cannot accidentally use it
    as an actuator callback without writing a separate, explicit integration.
    """

    shadow_only = True

    def __init__(
        self,
        settings: PLCSettings | None = None,
        *,
        cause_effect_path: str | Path = DEFAULT_CAUSE_EFFECT_PATH,
        st_path: str | Path = DEFAULT_ST_PATH,
    ) -> None:
        self.settings = settings or PLCSettings()
        self.controller_id, self.causes = _load_cause_effect(cause_effect_path)
        validate_plc_traceability(cause_effect_path, st_path)
        plant = load_reference_plant()
        fdv = next(
            actuator
            for actuator in plant["actuators"]
            if actuator["kind"] == "flow_diversion_valve"
        )
        if fdv["fail_position"] != "DIVERT":
            raise PLCLogicError("reference P&ID FDV is not fail-divert")
        self.reset()

    def reset(self) -> None:
        self.state = PLCState.OFF
        self.scan_index = 0
        self.state_elapsed_s = 0.0
        self.startup_elapsed_s = 0.0
        self.forward_confirmation_elapsed_s = 0.0
        self.cip_elapsed_s = 0.0
        self.trip_latched = False
        self.trip_acknowledged = False
        self.latched_trip_causes: set[str] = set()

    def _input_image(self, row: Mapping[str, object]) -> PLCInputs:
        if not isinstance(row, Mapping):
            raise PLCLogicError("PLC scan row must be a mapping")
        cfg = self.settings
        dt_s = _optional_number(row, "step_dt_s", cfg.scan_time_s)
        if dt_s <= 0.0:
            raise PLCLogicError("PLC step_dt_s must be positive")
        time_s = _optional_number(
            row,
            "campaign_time_s" if "campaign_time_s" in row else "time_s",
            (self.scan_index + 1) * dt_s,
        )
        safety_temp = _required_number(row, "safety_temp_sensor_c")
        residence = _required_number(row, "estimated_fastest_residence_time_s")
        measured_flow = _required_number(row, "measured_flow_l_h")
        maximum_flow = _required_number(row, "maximum_safe_flow_l_h")
        measured_dp = _required_number(row, "measured_differential_pressure_bar")
        disagreement_c = _required_number(row, "sensor_disagreement_c")
        leak_signal = _required_number(row, "leak_detector_signal_fraction")
        fdv_feedback = _required_number(row, "fdv_position_feedback")
        interface = _required_number(row, "restart_product_interface_signal_fraction")
        power_good = _required_bool(row, "power_good_signal")
        quality_ok = _required_bool(row, "temperature_sensor_quality_ok")
        cip_active = _required_bool(row, "cip_cycle_active")
        cip_release = _required_bool(row, "cip_release_permissive")
        fdv_mismatch = _required_bool(row, "alarm_fdv_mismatch")
        leak_alarm = _required_bool(row, "alarm_regenerator_leak")
        for name, value in (
            ("measured_flow_l_h", measured_flow),
            ("maximum_safe_flow_l_h", maximum_flow),
            ("estimated_fastest_residence_time_s", residence),
            ("leak_detector_signal_fraction", leak_signal),
        ):
            if value < 0.0:
                raise PLCLogicError(f"PLC input {name!r} must be non-negative")
        if not 0.0 <= fdv_feedback <= 1.0:
            raise PLCLogicError("fdv_position_feedback must be in [0, 1]")
        if not 0.0 <= interface <= 1.0:
            raise PLCLogicError("restart_product_interface_signal_fraction must be in [0, 1]")
        if maximum_flow <= 0.0:
            raise PLCLogicError("maximum_safe_flow_l_h must be positive")
        return PLCInputs(
            time_s=time_s,
            step_dt_s=dt_s,
            operator_start=_optional_bool(row, "operator_start", True),
            operator_stop=_optional_bool(row, "operator_stop", False),
            trip_ack=_optional_bool(row, "trip_ack", False),
            trip_reset=_optional_bool(row, "trip_reset", False),
            emergency_stop=_optional_bool(row, "emergency_stop", False),
            power_good=power_good,
            temperature_sensor_quality_ok=quality_ok,
            cip_active=cip_active,
            cip_release_permissive=cip_release,
            safety_temperature_c=safety_temp,
            estimated_fastest_residence_time_s=residence,
            measured_flow_l_h=measured_flow,
            maximum_safe_flow_l_h=maximum_flow,
            measured_differential_pressure_bar=measured_dp,
            sensor_disagreement_c=disagreement_c,
            leak_detector_signal_fraction=leak_signal,
            fdv_position_feedback=fdv_feedback,
            product_interface_fraction=interface,
            fdv_mismatch=fdv_mismatch,
            regenerator_leak=(leak_alarm or leak_signal >= cfg.leak_alarm_fraction),
            low_temperature=(
                safety_temp
                < cfg.diversion_threshold_c + cfg.forward_temperature_margin_c
            ),
            low_holding_time=(residence < cfg.minimum_holding_time_s),
            high_flow=(measured_flow > maximum_flow),
            low_differential_pressure=(
                measured_dp < cfg.required_differential_pressure_bar
            ),
            sensor_disagreement=(disagreement_c > cfg.sensor_disagreement_limit_c),
            plant_mode=str(row.get("plant_mode", "UNKNOWN")),
        )

    def _cip_recipe_state(self) -> CIPRecipeState:
        cfg = self.settings
        elapsed = self.cip_elapsed_s
        if elapsed < cfg.cip_pre_rinse_s:
            return CIPRecipeState.PRE_RINSE
        elapsed -= cfg.cip_pre_rinse_s
        if elapsed < cfg.cip_caustic_s:
            return CIPRecipeState.CAUSTIC
        elapsed -= cfg.cip_caustic_s
        if elapsed < cfg.cip_intermediate_rinse_s:
            return CIPRecipeState.INTERMEDIATE_RINSE
        elapsed -= cfg.cip_intermediate_rinse_s
        if elapsed < cfg.cip_acid_s:
            return CIPRecipeState.ACID
        elapsed -= cfg.cip_acid_s
        if elapsed < cfg.cip_final_rinse_s:
            return CIPRecipeState.FINAL_RINSE
        return CIPRecipeState.COMPLETE

    @staticmethod
    def _output_image(state: PLCState, recipe: CIPRecipeState) -> PLCOutputs:
        if state in {PLCState.STARTUP, PLCState.RECIRCULATE, PLCState.FORWARD}:
            return PLCOutputs(
                feed_pump_command=True,
                booster_pump_command=True,
                steam_enable=True,
                fdv_command_forward=(state == PLCState.FORWARD),
                fdv_command_divert=(state != PLCState.FORWARD),
            )
        if state == PLCState.CIP:
            rinse = recipe in {
                CIPRecipeState.PRE_RINSE,
                CIPRecipeState.INTERMEDIATE_RINSE,
                CIPRecipeState.FINAL_RINSE,
            }
            return PLCOutputs(
                cip_pump_command=(recipe != CIPRecipeState.COMPLETE),
                steam_enable=recipe in {CIPRecipeState.CAUSTIC, CIPRecipeState.ACID},
                fdv_command_forward=False,
                fdv_command_divert=True,
                cip_rinse_valve=rinse,
                cip_alkali_valve=(recipe == CIPRecipeState.CAUSTIC),
                cip_acid_valve=(recipe == CIPRecipeState.ACID),
                cip_drain_valve=(rinse or recipe == CIPRecipeState.COMPLETE),
            )
        # OFF and TRIP both de-energize every command.  DIVERT is represented
        # positively so the fail-safe intent remains explicit in the I/O image.
        return PLCOutputs(fdv_command_forward=False, fdv_command_divert=True)

    def scan(self, row: Mapping[str, object]) -> dict[str, object]:
        """Evaluate one shadow scan and return state, causes and explicit I/O.

        The input row is never modified.  Any missing/non-finite required signal
        fails closed by raising ``PLCLogicError`` rather than fabricating a safe
        process measurement.
        """

        inputs = self._input_image(row)
        previous_state = self.state
        active = [definition for definition in self.causes if _cause_active(inputs, definition)]
        active_trip = [item for item in active if item.classification == "TRIP"]
        active_divert = [item for item in active if item.classification == "DIVERT"]

        if active_trip:
            self.trip_latched = True
            self.trip_acknowledged = False
            self.latched_trip_causes.update(item.cause_id for item in active_trip)

        if self.trip_latched:
            self.state = PLCState.TRIP
            self.forward_confirmation_elapsed_s = 0.0
            self.startup_elapsed_s = 0.0
            if inputs.trip_ack:
                self.trip_acknowledged = True
            if inputs.trip_reset and self.trip_acknowledged and not active_trip:
                self.trip_latched = False
                self.trip_acknowledged = False
                self.latched_trip_causes.clear()
                self.state = PLCState.OFF
        elif inputs.operator_stop or not inputs.operator_start:
            self.state = PLCState.OFF
            self.startup_elapsed_s = 0.0
            self.forward_confirmation_elapsed_s = 0.0
            self.cip_elapsed_s = 0.0
        elif inputs.cip_active:
            if self.state != PLCState.CIP:
                self.cip_elapsed_s = 0.0
                self.forward_confirmation_elapsed_s = 0.0
            self.state = PLCState.CIP
            self.cip_elapsed_s += inputs.step_dt_s
        else:
            if self.state in {PLCState.OFF, PLCState.CIP, PLCState.TRIP}:
                self.state = PLCState.STARTUP
                self.startup_elapsed_s = 0.0
                self.forward_confirmation_elapsed_s = 0.0
                self.cip_elapsed_s = 0.0
            if self.state == PLCState.STARTUP:
                self.startup_elapsed_s += inputs.step_dt_s
                if self.startup_elapsed_s + 1e-12 >= self.settings.startup_delay_s:
                    self.state = PLCState.RECIRCULATE
                    self.forward_confirmation_elapsed_s = 0.0
            elif self.state == PLCState.FORWARD and active_divert:
                self.state = PLCState.RECIRCULATE
                self.forward_confirmation_elapsed_s = 0.0
            elif self.state == PLCState.RECIRCULATE:
                if active_divert:
                    self.forward_confirmation_elapsed_s = 0.0
                else:
                    self.forward_confirmation_elapsed_s += inputs.step_dt_s
                    if (
                        self.forward_confirmation_elapsed_s + 1e-12
                        >= self.settings.forward_confirmation_s
                    ):
                        self.state = PLCState.FORWARD

        if self.state == PLCState.CIP:
            recipe = self._cip_recipe_state()
        else:
            recipe = CIPRecipeState.IDLE
        output_image = self._output_image(self.state, recipe)

        # A final independent assertion protects the defining fail-divert
        # invariant even if future state/output logic is edited incorrectly.
        if self.state != PLCState.FORWARD and output_image.fdv_command_forward:
            raise AssertionError("reference PLC violated fail-divert invariant")
        if output_image.fdv_command_forward == output_image.fdv_command_divert:
            raise AssertionError("reference PLC emitted an ambiguous FDV command")

        if self.state == previous_state:
            self.state_elapsed_s += inputs.step_dt_s
        else:
            self.state_elapsed_s = 0.0
        self.scan_index += 1
        forward_permissive = not active_trip and not active_divert and not inputs.cip_active
        result: dict[str, object] = {
            "controller_id": self.controller_id,
            "shadow_only": True,
            "assurance_note": (
                "Reference PLC shadow output only; it does not drive the HTST "
                "simulator or physical equipment."
            ),
            "scan_index": self.scan_index,
            "time_s": inputs.time_s,
            "previous_state": previous_state.value,
            "plc_state": self.state.value,
            "state_changed": int(previous_state != self.state),
            "state_elapsed_s": self.state_elapsed_s,
            "cip_recipe_state": recipe.value,
            "cip_elapsed_s": self.cip_elapsed_s,
            "forward_permissive": int(forward_permissive),
            "forward_confirmation_elapsed_s": self.forward_confirmation_elapsed_s,
            "trip_latched": int(self.trip_latched),
            "trip_acknowledged": int(self.trip_acknowledged),
            "active_causes": "|".join(item.cause_id for item in active),
            "active_trip_causes": "|".join(item.cause_id for item in active_trip),
            "active_divert_causes": "|".join(item.cause_id for item in active_divert),
            "latched_trip_causes": "|".join(sorted(self.latched_trip_causes)),
            "input_image": asdict(inputs),
            "output_image": asdict(output_image),
        }
        return result


def run_plc_shadow(
    rows: Iterable[Mapping[str, object]],
    plc: ReferencePLC | None = None,
) -> list[dict[str, object]]:
    """Run ordered simulator/campaign rows through one persistent PLC shadow."""

    controller = plc or ReferencePLC()
    return [controller.scan(row) for row in rows]


__all__ = [
    "CAUSE_EFFECT_SCHEMA_VERSION",
    "CIPRecipeState",
    "DEFAULT_CAUSE_EFFECT_PATH",
    "DEFAULT_ST_PATH",
    "PLCInputs",
    "PLCLogicError",
    "PLCOutputs",
    "PLCSettings",
    "PLCState",
    "ReferencePLC",
    "SUPPORTED_CAUSE_RULES",
    "run_plc_shadow",
    "validate_plc_traceability",
]
