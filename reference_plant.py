#!/usr/bin/env python3
"""Machine-readable reference P&ID contract for the HTST shadow simulator.

The topology in :mod:`reference_pid.json` is deliberately a public-reference
research plant.  It is not an as-built drawing, does not replace line walkdown
or design review, and must never be used to operate physical equipment.  This
module validates structural traceability only: tags, directed process paths,
simulator mappings, actuator fail positions and I/O address uniqueness.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PID = HERE / "reference_pid.json"
REFERENCE_PLANT_SCHEMA_VERSION = "1.0.0"

_TAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
_FAIL_POSITIONS = {"CLOSED", "DIVERT", "OPEN", "STOPPED"}

# This is an explicit adapter contract, not introspection of model.py.  A field
# can appear here even when it is an oracle in the simulator; its P&ID mapping
# does not make that field an online ML feature or a verified plant signal.
KNOWN_SIMULATOR_FIELDS = frozenset(
    {
        "balance_tank_volume_l",
        "booster_pump_factor",
        "booster_pump_speed_fraction",
        "cip_chemical_concentration_pct",
        "cip_conductivity_proxy_ms_cm",
        "cip_cycle_active",
        "cip_ph_proxy",
        "control_temp_sensor_c",
        "fdv_command_forward",
        "fdv_position",
        "fdv_position_feedback",
        "flow_l_h",
        "inlet_temp_c",
        "leak_detector_signal_fraction",
        "measured_differential_pressure_bar",
        "measured_flow_l_h",
        "pasteurized_pressure_sensor_bar",
        "power_good_signal",
        "preheat_temp_sensor_c",
        "product_temp_sensor_c",
        "raw_pressure_sensor_bar",
        "restart_product_interface_signal_fraction",
        "safety_temp_sensor_c",
        "steam_valve",
        "temperature_sensor_quality_ok",
        "transition_drained_l",
    }
)

KNOWN_PLC_OUTPUT_FIELDS = frozenset(
    {
        "booster_pump_command",
        "cip_acid_valve",
        "cip_alkali_valve",
        "cip_drain_valve",
        "cip_pump_command",
        "fdv_command_forward",
        "feed_pump_command",
        "steam_enable",
    }
)


class ReferencePlantError(ValueError):
    """Raised when the reference P&ID violates its structural contract."""


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferencePlantError(f"{context} must be an object")
    return value


def _items(value: object, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ReferencePlantError(f"{context} must be a non-empty array")
    return [_mapping(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _text(item: Mapping[str, Any], name: str, context: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ReferencePlantError(f"{context}.{name} must be a non-empty string")
    return value.strip()


def _tag(item: Mapping[str, Any], context: str) -> str:
    value = _text(item, "tag", context)
    if not _TAG_PATTERN.fullmatch(value):
        raise ReferencePlantError(f"{context}.tag has invalid syntax: {value!r}")
    return value


def _finite_nonnegative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferencePlantError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ReferencePlantError(f"{context} must be finite and non-negative")
    return result


def _reject_constant(value: str) -> None:
    raise ReferencePlantError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferencePlantError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def validate_reference_plant(plant: Mapping[str, Any]) -> dict[str, object]:
    """Validate a reference plant and return a compact validation report.

    Validation is intentionally strict and deterministic.  In particular,
    every tag is globally unique, every instrument and actuator is mapped to a
    declared simulator field, every actuator has a known fail position, every
    I/O address is globally unique, and every consecutive edge of each required
    path exists in the directed line list.
    """

    root = _mapping(plant, "reference plant")
    required_top = {
        "schema_version",
        "plant_id",
        "title",
        "model_status",
        "shadow_only",
        "boundaries",
        "equipment",
        "actuators",
        "lines",
        "instruments",
        "required_paths",
    }
    missing = required_top - set(root)
    if missing:
        raise ReferencePlantError(f"reference plant is missing keys: {sorted(missing)}")
    if root["schema_version"] != REFERENCE_PLANT_SCHEMA_VERSION:
        raise ReferencePlantError(
            "unsupported reference plant schema_version: "
            f"{root['schema_version']!r}"
        )
    _text(root, "plant_id", "reference plant")
    _text(root, "title", "reference plant")
    status = _text(root, "model_status", "reference plant").lower()
    if "unvalidated" not in status or "reference" not in status:
        raise ReferencePlantError(
            "model_status must identify the topology as an unvalidated reference"
        )
    if root["shadow_only"] is not True:
        raise ReferencePlantError("reference plant must declare shadow_only=true")

    collections = {
        "boundaries": _items(root["boundaries"], "boundaries"),
        "equipment": _items(root["equipment"], "equipment"),
        "actuators": _items(root["actuators"], "actuators"),
        "lines": _items(root["lines"], "lines"),
        "instruments": _items(root["instruments"], "instruments"),
    }
    tags: dict[str, str] = {}
    category_tags: dict[str, list[str]] = {}
    for category, values in collections.items():
        category_tags[category] = []
        for index, item in enumerate(values):
            context = f"{category}[{index}]"
            tag = _tag(item, context)
            if tag in tags:
                raise ReferencePlantError(
                    f"duplicate tag {tag!r} in {context}; first used by {tags[tag]}"
                )
            tags[tag] = context
            category_tags[category].append(tag)

    node_tags = set(category_tags["boundaries"])
    node_tags.update(category_tags["equipment"])
    node_tags.update(category_tags["actuators"])
    line_tags = set(category_tags["lines"])
    instrument_tags = set(category_tags["instruments"])

    for index, item in enumerate(collections["boundaries"]):
        _text(item, "kind", f"boundaries[{index}]")
        _text(item, "description", f"boundaries[{index}]")

    for index, item in enumerate(collections["equipment"]):
        _text(item, "kind", f"equipment[{index}]")
        _text(item, "description", f"equipment[{index}]")
        _text(item, "simulator_component", f"equipment[{index}]")

    io_owners: dict[str, str] = {}

    def register_io(address: object, owner: str) -> str:
        if not isinstance(address, str) or not address.strip():
            raise ReferencePlantError(f"{owner} must be a non-empty I/O address")
        normalized = address.strip()
        if normalized in io_owners:
            raise ReferencePlantError(
                f"duplicate I/O address {normalized!r}: {io_owners[normalized]} and {owner}"
            )
        io_owners[normalized] = owner
        return normalized

    fdv_count = 0
    for index, actuator in enumerate(collections["actuators"]):
        context = f"actuators[{index}]"
        tag = category_tags["actuators"][index]
        _text(actuator, "kind", context)
        _text(actuator, "description", context)
        field = _text(actuator, "simulator_field", context)
        if field not in KNOWN_SIMULATOR_FIELDS:
            raise ReferencePlantError(
                f"{context}.simulator_field is not in the adapter contract: {field!r}"
            )
        command = _text(actuator, "command_field", context)
        if command not in KNOWN_PLC_OUTPUT_FIELDS:
            raise ReferencePlantError(
                f"{context}.command_field is not a reference PLC output: {command!r}"
            )
        fail_position = _text(actuator, "fail_position", context).upper()
        if fail_position not in _FAIL_POSITIONS:
            raise ReferencePlantError(
                f"{context}.fail_position is unsupported: {fail_position!r}"
            )
        register_io(actuator.get("command_io"), f"{tag}.command_io")
        feedback_field = actuator.get("feedback_field")
        feedback_io = actuator.get("feedback_io")
        if (feedback_field is None) != (feedback_io is None):
            raise ReferencePlantError(
                f"{context} must declare feedback_field and feedback_io together"
            )
        if feedback_field is not None:
            if not isinstance(feedback_field, str) or feedback_field not in KNOWN_SIMULATOR_FIELDS:
                raise ReferencePlantError(
                    f"{context}.feedback_field is not in the adapter contract"
                )
            register_io(feedback_io, f"{tag}.feedback_io")
        if actuator["kind"] == "flow_diversion_valve":
            fdv_count += 1
            if fail_position != "DIVERT":
                raise ReferencePlantError(
                    "the reference flow-diversion valve must fail to DIVERT"
                )
    if fdv_count != 1:
        raise ReferencePlantError(
            f"reference plant must contain exactly one flow_diversion_valve, got {fdv_count}"
        )

    directed_edges: set[tuple[str, str]] = set()
    for index, line in enumerate(collections["lines"]):
        context = f"lines[{index}]"
        source = _text(line, "from", context)
        destination = _text(line, "to", context)
        if source not in node_tags or destination not in node_tags:
            raise ReferencePlantError(
                f"{context} endpoint is not a boundary, equipment or actuator tag: "
                f"{source!r}->{destination!r}"
            )
        if source == destination:
            raise ReferencePlantError(f"{context} cannot be a self-loop")
        edge = (source, destination)
        if edge in directed_edges:
            raise ReferencePlantError(
                f"duplicate directed connection {source!r}->{destination!r}"
            )
        directed_edges.add(edge)
        _text(line, "service", context)
        _finite_nonnegative(line.get("nominal_holdup_l"), f"{context}.nominal_holdup_l")

    for actuator_tag in category_tags["actuators"]:
        incoming = any(destination == actuator_tag for _, destination in directed_edges)
        outgoing = any(source == actuator_tag for source, _ in directed_edges)
        if not incoming or not outgoing:
            raise ReferencePlantError(
                f"actuator {actuator_tag!r} must have both incoming and outgoing lines"
            )

    installable = node_tags | line_tags | instrument_tags
    for index, instrument in enumerate(collections["instruments"]):
        context = f"instruments[{index}]"
        tag = category_tags["instruments"][index]
        _text(instrument, "kind", context)
        location = _text(instrument, "installed_on", context)
        if location not in installable:
            raise ReferencePlantError(
                f"{context}.installed_on references unknown tag {location!r}"
            )
        field = _text(instrument, "simulator_field", context)
        if field not in KNOWN_SIMULATOR_FIELDS:
            raise ReferencePlantError(
                f"{context}.simulator_field is not in the adapter contract: {field!r}"
            )
        register_io(instrument.get("io_address"), f"{tag}.io_address")
        _text(instrument, "unit", context)

    paths = _items(root["required_paths"], "required_paths")
    path_ids: set[str] = set()
    validated_paths: list[str] = []
    for index, path in enumerate(paths):
        context = f"required_paths[{index}]"
        path_id = _text(path, "path_id", context)
        if path_id in path_ids:
            raise ReferencePlantError(f"duplicate required path_id {path_id!r}")
        path_ids.add(path_id)
        _text(path, "description", context)
        nodes = path.get("nodes")
        if (
            not isinstance(nodes, list)
            or len(nodes) < 2
            or not all(isinstance(node, str) and node for node in nodes)
        ):
            raise ReferencePlantError(f"{context}.nodes must contain at least two tags")
        unknown = [node for node in nodes if node not in node_tags]
        if unknown:
            raise ReferencePlantError(
                f"{context}.nodes references unknown tags: {unknown}"
            )
        missing_edges = [
            f"{left}->{right}"
            for left, right in zip(nodes, nodes[1:])
            if (left, right) not in directed_edges
        ]
        if missing_edges:
            raise ReferencePlantError(
                f"required path {path_id!r} has missing directed edges: {missing_edges}"
            )
        validated_paths.append(path_id)

    mandatory_paths = {"RAW_TO_PRODUCT", "FDV_DIVERT_RETURN", "CIP_RECIRCULATION"}
    if not mandatory_paths.issubset(path_ids):
        raise ReferencePlantError(
            f"required_paths lacks mandatory paths: {sorted(mandatory_paths - path_ids)}"
        )

    return {
        "valid": True,
        "plant_id": root["plant_id"],
        "schema_version": root["schema_version"],
        "shadow_only": True,
        "tag_count": len(tags),
        "node_count": len(node_tags),
        "line_count": len(line_tags),
        "instrument_count": len(instrument_tags),
        "actuator_count": len(category_tags["actuators"]),
        "io_count": len(io_owners),
        "validated_paths": validated_paths,
    }


def load_reference_plant(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the bundled reference plant or an explicit JSON file."""

    source = DEFAULT_REFERENCE_PID if path is None else Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ReferencePlantError:
        raise
    except OSError as exc:
        raise ReferencePlantError(f"cannot read reference P&ID {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReferencePlantError(f"invalid JSON in reference P&ID {source}: {exc}") from exc
    plant = dict(_mapping(value, "reference plant"))
    validate_reference_plant(plant)
    return plant


def _label(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_mermaid(plant: Mapping[str, Any] | None = None) -> str:
    """Render a GitHub-compatible Mermaid view of the validated reference P&ID.

    The returned value starts with ``flowchart LR`` and intentionally omits the
    Markdown fence so callers can embed it in README-style documents or test it
    independently.
    """

    value = load_reference_plant() if plant is None else dict(plant)
    validate_reference_plant(value)
    categories: Sequence[tuple[str, str]] = (
        ("boundaries", "boundary"),
        ("equipment", "equipment"),
        ("actuators", "actuator"),
        ("instruments", "instrument"),
    )
    node_ids: dict[str, str] = {}
    class_nodes: dict[str, list[str]] = {name: [] for _, name in categories}
    lines = [
        "flowchart LR",
        "    %% Unvalidated reference/shadow topology; never an as-built control drawing.",
    ]
    counter = 0
    for collection_name, class_name in categories:
        for raw in value[collection_name]:
            item = _mapping(raw, collection_name)
            tag = str(item["tag"])
            node_id = f"N{counter:03d}"
            counter += 1
            node_ids[tag] = node_id
            class_nodes[class_name].append(node_id)
            description = str(item.get("description", item.get("kind", "")))
            label = f"{_label(tag)}<br/>{_label(description)}"
            if item.get("kind") == "flow_diversion_valve":
                lines.append(f'    {node_id}{{"{label}"}}')
            elif class_name == "instrument":
                lines.append(f'    {node_id}(("{_label(tag)}"))')
            else:
                lines.append(f'    {node_id}["{label}"]')

    for raw in value["lines"]:
        line = _mapping(raw, "line")
        source = node_ids[str(line["from"])]
        destination = node_ids[str(line["to"])]
        edge_label = _label(f"{line['tag']} · {line['service']}")
        lines.append(f'    {source} -->|"{edge_label}"| {destination}')

    for raw in value["instruments"]:
        instrument = _mapping(raw, "instrument")
        location = str(instrument["installed_on"])
        # Instruments can be installed on a line.  In that case attach their
        # visual trace to the line's downstream process node.
        if location in node_ids:
            target = node_ids[location]
        else:
            line = next(item for item in value["lines"] if item["tag"] == location)
            target = node_ids[str(line["to"])]
        lines.append(
            f"    {node_ids[str(instrument['tag'])]} -. "
            f'"{_label(instrument["simulator_field"])}" .-> {target}'
        )

    lines.extend(
        [
            "    classDef boundary fill:#f5f5f5,stroke:#616161,color:#212121;",
            "    classDef equipment fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;",
            "    classDef actuator fill:#fff3e0,stroke:#ef6c00,color:#e65100;",
            "    classDef instrument fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;",
        ]
    )
    for class_name, members in class_nodes.items():
        if members:
            lines.append(f"    class {','.join(members)} {class_name};")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_REFERENCE_PID",
    "KNOWN_PLC_OUTPUT_FIELDS",
    "KNOWN_SIMULATOR_FIELDS",
    "REFERENCE_PLANT_SCHEMA_VERSION",
    "ReferencePlantError",
    "load_reference_plant",
    "render_mermaid",
    "validate_reference_plant",
]
