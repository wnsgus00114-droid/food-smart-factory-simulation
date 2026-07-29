"""Leakage-safe P&ID adapter and transport-delay graph contract.

Only observables admitted by ``ml_contract.json`` can be attached to graph
nodes.  P&ID connectivity is a structural prior; it does not turn simulator
truth fields into deployable ML inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml_pipeline_common import BASE_DIR, feature_fields, load_contract
from reference_plant import load_reference_plant
from sensor_calibration import combined_standard_uncertainty


GRAPH_VERSION = "0.1.0"

# Line-mounted instruments are attached to the downstream process node, which
# is the same convention used by reference_plant.render_mermaid().  Derived
# observables are explicitly attached to the physical asset that produces or
# consumes their information.
FEATURE_NODE_MAP: dict[str, str] = {
    "control_temp_sensor_c": "HT-101",
    "safety_temp_sensor_c": "HT-101",
    "measured_flow_l_h": "P-102",
    "steam_valve": "TV-101",
    "fdv_command_forward": "FDV-101",
    "fdv_command_position": "FDV-101",
    "fdv_position_feedback": "FDV-101",
    "fdv_position_error": "FDV-101",
    "product_temp_sensor_c": "PRODUCT-OUTLET",
    "raw_pressure_sensor_bar": "HX-101-A",
    "pasteurized_pressure_sensor_bar": "FDV-101",
    "measured_differential_pressure_bar": "P-102",
    "booster_pump_speed_fraction": "P-102",
    "leak_detector_signal_fraction": "HX-101-A",
    "preheat_temp_sensor_c": "HX-101-B",
    "sensor_disagreement_c": "HT-101",
    "estimated_residence_time_s": "HT-101",
    "estimated_fastest_residence_time_s": "HT-101",
    "maximum_safe_flow_l_h": "HT-101",
    "cip_chemical_concentration_pct": "CIP-201",
    "cip_cycle_active": "CIP-201",
    "cip_conductivity_proxy_ms_cm": "CIP-201",
    "cip_ph_proxy": "CIP-201",
    "cip_release_permissive": "CIP-201",
    "post_cip_conductivity_proxy_ms_cm": "CIP-201",
    "post_cip_ph_proxy": "CIP-201",
    "restart_product_interface_signal_fraction": "HX-101-C",
    "power_good_signal": "P-101",
    "temperature_sensor_quality_ok": "HT-101",
}

# Advective features are carried with product/CIP material.  Hydraulic and
# control features use the instantaneous relation instead of an artificial
# V/Q delay.  A feature belongs to one primary relation; both streams are
# fused after graph propagation.
FEATURE_RELATION_MAP: dict[str, str] = {
    "control_temp_sensor_c": "advective",
    "safety_temp_sensor_c": "advective",
    "product_temp_sensor_c": "advective",
    "preheat_temp_sensor_c": "advective",
    "cip_chemical_concentration_pct": "advective",
    "cip_conductivity_proxy_ms_cm": "advective",
    "cip_ph_proxy": "advective",
    "post_cip_conductivity_proxy_ms_cm": "advective",
    "post_cip_ph_proxy": "advective",
    "restart_product_interface_signal_fraction": "advective",
    "measured_flow_l_h": "hydraulic_control",
    "steam_valve": "hydraulic_control",
    "fdv_command_forward": "hydraulic_control",
    "fdv_command_position": "hydraulic_control",
    "fdv_position_feedback": "hydraulic_control",
    "fdv_position_error": "hydraulic_control",
    "raw_pressure_sensor_bar": "hydraulic_control",
    "pasteurized_pressure_sensor_bar": "hydraulic_control",
    "measured_differential_pressure_bar": "hydraulic_control",
    "booster_pump_speed_fraction": "hydraulic_control",
    "leak_detector_signal_fraction": "hydraulic_control",
    "sensor_disagreement_c": "hydraulic_control",
    "estimated_residence_time_s": "hydraulic_control",
    "estimated_fastest_residence_time_s": "hydraulic_control",
    "maximum_safe_flow_l_h": "hydraulic_control",
    "cip_cycle_active": "hydraulic_control",
    "cip_release_permissive": "hydraulic_control",
    "power_good_signal": "hydraulic_control",
    "temperature_sensor_quality_ok": "hydraulic_control",
}

CONTROL_FIELDS = (
    "measured_flow_l_h",
    "cip_cycle_active",
    "fdv_position_feedback",
    "steam_valve",
    "power_good_signal",
    "cip_chemical_concentration_pct",
)

# The first three gate groups are material paths.  Shared edges L-004..L-007
# are active in both production and CIP.  Gate definitions never use
# plant_mode, scenario, true valve position, or downstream oracle outcomes.
EDGE_GATE_KIND: dict[str, str] = {
    **{f"L-{number:03d}": "production" for number in range(1, 4)},
    **{f"L-{number:03d}": "common" for number in range(4, 8)},
    **{f"L-{number:03d}": "forward" for number in range(8, 11)},
    "L-011": "divert",
    "L-201": "cip",
    "L-202": "cip",
    "L-203": "cip",
    # D1 lacks drain and individual acid/alkali valve commands.  Keeping these
    # edges closed is more honest than manufacturing a route label.
    "L-204": "unobserved",
    "L-205": "unobserved",
    "U-101": "steam",
    "U-102": "steam",
    "U-103": "cooling",
    "C-201": "unobserved",
    "C-202": "unobserved",
    "C-203": "unobserved",
    "C-204": "unobserved",
}

UTILITY_SERVICES = frozenset({"heating_utility", "cooling_utility"})


@dataclass(frozen=True)
class GraphNode:
    tag: str
    kind: str
    category: str


@dataclass(frozen=True)
class GraphEdge:
    tag: str
    source: int
    destination: int
    service: str
    nominal_holdup_l: float
    gate_kind: str
    delay_kind: str
    advective: bool


@dataclass(frozen=True)
class ProcessGraph:
    version: str
    plant_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    features: tuple[str, ...]
    feature_node_indices: tuple[int, ...]
    feature_relations: tuple[str, ...]
    control_feature_indices: tuple[int, ...]
    sensor_standard_uncertainty_raw: tuple[float, ...]
    sensor_uncertainty_known: tuple[bool, ...]

    @property
    def node_tags(self) -> tuple[str, ...]:
        return tuple(node.tag for node in self.nodes)

    @property
    def edge_tags(self) -> tuple[str, ...]:
        return tuple(edge.tag for edge in self.edges)

    def edge_volumes_for_profile(self, profile: Mapping[str, object]) -> tuple[float, ...]:
        """Overlay simulator-exact inventories on the reference P&ID prior.

        L-005 carries the holding-tube dwell, L-007 the safety-sensor-to-FDV
        line, and L-008..L-010 proportional shares of the aggregate post-FDV
        queue.  Other material holdups remain explicitly nominal P&ID priors.
        """

        def positive(name: str) -> float:
            value = profile.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(f"profile {name} must be numeric")
            result = float(value)
            if not math.isfinite(result) or result <= 0.0:
                raise ValueError(f"profile {name} must be finite and positive")
            return result

        nominal_flow_l_h = positive("nominal_flow_l_h")
        flow_l_s = nominal_flow_l_h / 3600.0
        holding_volume = flow_l_s * positive("nominal_holding_time_s")
        fdv_volume = flow_l_s * positive("sensor_to_fdv_delay_s")
        post_volume = flow_l_s * positive("post_fdv_residence_time_s")
        post_tags = ("L-008", "L-009", "L-010")
        post_prior_total = sum(
            edge.nominal_holdup_l for edge in self.edges if edge.tag in post_tags
        )
        volumes: list[float] = []
        for edge in self.edges:
            if edge.tag == "L-005":
                value = holding_volume
            elif edge.tag == "L-007":
                value = fdv_volume
            elif edge.tag in post_tags:
                value = post_volume * edge.nominal_holdup_l / post_prior_total
            else:
                value = edge.nominal_holdup_l
            volumes.append(value)
        return tuple(volumes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_version": self.version,
            "plant_id": self.plant_id,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
            "features": list(self.features),
            "feature_node_indices": list(self.feature_node_indices),
            "feature_relations": list(self.feature_relations),
            "control_fields": list(CONTROL_FIELDS),
            "sensor_standard_uncertainty_raw": list(
                self.sensor_standard_uncertainty_raw
            ),
            "sensor_uncertainty_known": list(self.sensor_uncertainty_known),
        }


def _node_records(plant: Mapping[str, Any]) -> tuple[list[GraphNode], dict[str, int]]:
    nodes: list[GraphNode] = []
    for category in ("boundaries", "equipment", "actuators"):
        for item in plant[category]:
            nodes.append(
                GraphNode(
                    tag=str(item["tag"]),
                    kind=str(item["kind"]),
                    category=category[:-1],
                )
            )
    indexes = {node.tag: index for index, node in enumerate(nodes)}
    if len(indexes) != len(nodes):
        raise ValueError("process graph contains duplicate node tags")
    return nodes, indexes


def _delay_kind(tag: str, service: str) -> str:
    if tag in {"L-005", "L-007", "L-008", "L-009", "L-010"}:
        return "exact_fifo"
    if tag == "L-011":
        return "exact_one_step_return"
    if tag in {"L-201", "L-202", "L-203"}:
        return "perfect_mix_inventory"
    if service in UTILITY_SERVICES:
        return "utility_or_control"
    if EDGE_GATE_KIND.get(tag) == "unobserved":
        return "unobserved"
    return "nominal_pid_prior"


def _catalog_uncertainties(
    plant: Mapping[str, Any], features: Sequence[str], catalog_path: Path
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_tag = {str(item["tag"]): item for item in catalog["sensors"]}
    field_to_tag = {
        str(item["simulator_field"]): str(item["tag"])
        for item in plant["instruments"]
    }
    values: list[float] = []
    known: list[bool] = []
    for field in features:
        tag = field_to_tag.get(field)
        sensor = by_tag.get(tag or "")
        if sensor is None:
            values.append(0.0)
            known.append(False)
            continue
        components = {
            str(name): float(value)
            for name, value in sensor["calibration"][
                "uncertainty_components"
            ].items()
        }
        values.append(combined_standard_uncertainty(components))
        known.append(True)
    return tuple(values), tuple(known)


def build_process_graph(
    *,
    feature_set: str = "S3-context",
    pid_path: Path | None = None,
    contract_path: Path | None = None,
    sensor_catalog_path: Path | None = None,
) -> ProcessGraph:
    """Build the deterministic 22-node graph after enforcing the ML allowlist."""

    contract = load_contract(contract_path)
    features = tuple(feature_fields(contract, feature_set))
    forbidden = set(contract["always_forbidden_features"])
    overlap = forbidden & set(features)
    if overlap:
        raise ValueError(f"forbidden fields entered graph features: {sorted(overlap)}")
    missing_mapping = set(features) - set(FEATURE_NODE_MAP)
    missing_relation = set(features) - set(FEATURE_RELATION_MAP)
    if missing_mapping or missing_relation:
        raise ValueError(
            "graph adapter lacks mappings: "
            f"nodes={sorted(missing_mapping)}, relations={sorted(missing_relation)}"
        )
    # Context fields such as time_s and plant_mode are intentionally rejected,
    # even if a caller tries to add them outside the feature-set contract.
    context_only = {"time_s", "time_start_s", "plant_mode"}
    if context_only & set(features):
        raise ValueError("time and plant_mode are structural context, not model inputs")

    plant = load_reference_plant(pid_path)
    nodes, node_index = _node_records(plant)
    edges: list[GraphEdge] = []
    for raw in plant["lines"]:
        tag = str(raw["tag"])
        service = str(raw["service"])
        gate_kind = EDGE_GATE_KIND.get(tag)
        if gate_kind is None:
            raise ValueError(f"line {tag} has no observable route-gate contract")
        edges.append(
            GraphEdge(
                tag=tag,
                source=node_index[str(raw["from"])],
                destination=node_index[str(raw["to"])],
                service=service,
                nominal_holdup_l=float(raw["nominal_holdup_l"]),
                gate_kind=gate_kind,
                delay_kind=_delay_kind(tag, service),
                advective=service not in UTILITY_SERVICES,
            )
        )

    feature_node_indices = tuple(node_index[FEATURE_NODE_MAP[name]] for name in features)
    control_indices = tuple(features.index(field) for field in CONTROL_FIELDS)
    uncertainty, known = _catalog_uncertainties(
        plant,
        features,
        sensor_catalog_path or BASE_DIR / "sensor_catalog.json",
    )
    return ProcessGraph(
        version=GRAPH_VERSION,
        plant_id=str(plant["plant_id"]),
        nodes=tuple(nodes),
        edges=tuple(edges),
        features=features,
        feature_node_indices=feature_node_indices,
        feature_relations=tuple(FEATURE_RELATION_MAP[name] for name in features),
        control_feature_indices=control_indices,
        sensor_standard_uncertainty_raw=uncertainty,
        sensor_uncertainty_known=known,
    )


__all__ = [
    "CONTROL_FIELDS",
    "EDGE_GATE_KIND",
    "FEATURE_NODE_MAP",
    "FEATURE_RELATION_MAP",
    "GRAPH_VERSION",
    "GraphEdge",
    "GraphNode",
    "ProcessGraph",
    "build_process_graph",
]
