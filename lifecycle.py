#!/usr/bin/env python3
"""Deterministic multi-cycle lifecycle surrogate for the HTST reference plant.

The module deliberately separates reversible surface fouling from irreversible
damage.  It is a synthetic research model: its Weibull, damage, cleaning and
repair parameters are not fitted to a physical asset population.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


LIFECYCLE_MODEL_VERSION = "1.0.0"
POLICY_SCHEMA_VERSION = "1.0.0"
MODEL_STATUS = "unvalidated_engineering_assumption"
FAILURE_CAUSE_PRIORITY = (
    "DAMAGE_LIMIT",
    "FOULING_LIMIT",
    "WEIBULL_FAILURE",
)

TRACE_FIELDS = (
    "lifecycle_model_version",
    "run_id",
    "policy_id",
    "policy_hash",
    "seed",
    "cycle",
    "cycle_end_time_h",
    "asset_id",
    "component_type",
    "failure_mode",
    "cip_kind",
    "scheduled_action",
    "failures_in_cycle",
    "renewal_count",
    "service_interval_id",
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
    "median_competing_cause",
)

EVENT_FIELDS = (
    "lifecycle_model_version",
    "run_id",
    "event_id",
    "time_h",
    "cycle",
    "asset_id",
    "component_type",
    "event_type",
    "trigger_stage",
    "failure_cause",
    "maintenance_action",
    "cip_kind",
    "service_interval_id",
    "event_observed",
    "virtual_age_before_h",
    "virtual_age_after_h",
    "damage_before",
    "damage_after",
    "fouling_before",
    "fouling_after",
    "virtual_age_factor_q",
    "damage_factor",
    "fouling_factor",
)

SERVICE_INTERVAL_FIELDS = (
    "lifecycle_model_version",
    "run_id",
    "service_interval_id",
    "asset_id",
    "component_type",
    "interval_index",
    "generation",
    "start_time_h",
    "end_time_h",
    "duration_h",
    "start_cycle",
    "end_cycle",
    "start_action",
    "terminal_event_type",
    "event_observed",
    "failure_cause",
    "exact_rul_h",
    "rul_lower_bound_h",
    "start_virtual_age_h",
    "end_virtual_age_h",
    "start_damage",
    "end_damage",
    "start_fouling",
    "end_fouling",
    "is_last_interval",
)


class PolicyValidationError(ValueError):
    """Raised when a maintenance policy violates its strict JSON contract."""


@dataclass(frozen=True)
class CIPRecipe:
    name: str
    fouling_removal_fraction: float
    chemical_exposure_factor: float
    description: str


@dataclass(frozen=True)
class ComponentSpec:
    asset_id: str
    component_type: str
    failure_mode: str
    weibull_shape: float
    weibull_scale_h: float
    damage_rate_per_operating_h: float
    cip_damage_per_exposure: float
    fouling_rate_per_operating_h: float
    damage_limit: float
    fouling_limit: float
    priority: int


@dataclass(frozen=True)
class MaintenanceAction:
    asset_id: str | None
    every_cycles: int | None
    action: str
    virtual_age_factor: float
    damage_factor: float
    fouling_factor: float


@dataclass(frozen=True)
class MaintenancePolicy:
    schema_version: str
    policy_id: str
    description: str
    model_status: str
    seed: int
    default_cycles: int
    production_hours_per_cycle: float
    load_factor: float
    cip_pattern: tuple[str, ...]
    cip_recipes: tuple[CIPRecipe, ...]
    components: tuple[ComponentSpec, ...]
    scheduled_actions: tuple[MaintenanceAction, ...]
    corrective_action: MaintenanceAction
    sources: tuple[dict[str, str], ...]
    canonical_payload: dict[str, Any]

    @property
    def component_by_id(self) -> dict[str, ComponentSpec]:
        return {item.asset_id: item for item in self.components}

    @property
    def recipe_by_name(self) -> dict[str, CIPRecipe]:
        return {item.name: item for item in self.cip_recipes}

    @property
    def action_by_asset(self) -> dict[str, MaintenanceAction]:
        return {
            str(item.asset_id): item
            for item in self.scheduled_actions
            if item.asset_id is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.canonical_payload)


@dataclass
class ComponentState:
    spec: ComponentSpec
    virtual_age_h: float
    irreversible_damage: float
    reversible_fouling: float
    weibull_remaining_hazard: float
    renewal_count: int
    interval_index: int
    interval_start_time_h: float
    interval_start_cycle: int
    interval_start_action: str
    interval_start_virtual_age_h: float
    interval_start_damage: float
    interval_start_fouling: float

    @property
    def service_interval_id(self) -> str:
        return f"{self.spec.asset_id}-SI-{self.interval_index:04d}"


@dataclass(frozen=True)
class LifecycleResult:
    run_id: str
    policy_hash: str
    seed: int
    cycles: int
    trace: list[dict[str, object]]
    events: list[dict[str, object]]
    service_intervals: list[dict[str, object]]
    summary: dict[str, object]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PolicyValidationError(f"non-finite JSON number is forbidden: {value}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{context} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise PolicyValidationError(
            f"{context} keys differ from contract; missing={missing}, unknown={unknown}"
        )


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyValidationError(f"{context} must be a non-empty string")
    return value.strip()


def _number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyValidationError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyValidationError(f"{context} must be finite")
    if strictly_positive and result <= 0.0:
        raise PolicyValidationError(f"{context} must be > 0")
    if minimum is not None and result < minimum:
        raise PolicyValidationError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PolicyValidationError(f"{context} must be <= {maximum}")
    return result


def _integer(
    value: Any, context: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyValidationError(f"{context} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise PolicyValidationError(f"{context} must be {minimum}{suffix}")
    return value


def _normal_action_name(value: str) -> str:
    aliases = {
        "calibrate": "calibration",
        "calibration": "calibration",
        "minor_repair": "minor_repair",
        "overhaul": "overhaul",
        "replace": "replacement",
        "replacement": "replacement",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise PolicyValidationError(
            "maintenance action must be calibrate/calibration, minor_repair, "
            "overhaul, replace or replacement"
        ) from exc


def validate_policy(payload: Mapping[str, Any]) -> MaintenancePolicy:
    """Validate and canonicalize a maintenance-policy object."""

    raw = _mapping(payload, "policy")
    top_keys = {
        "schema_version",
        "policy_id",
        "description",
        "model_status",
        "seed",
        "default_cycles",
        "production_hours_per_cycle",
        "load_factor",
        "cip_pattern",
        "cip_recipes",
        "components",
        "scheduled_actions",
        "corrective_action",
        "sources",
    }
    _exact_keys(raw, top_keys, "policy")
    schema_version = _text(raw["schema_version"], "schema_version")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise PolicyValidationError(
            f"unsupported schema_version {schema_version!r}; expected {POLICY_SCHEMA_VERSION!r}"
        )
    model_status = _text(raw["model_status"], "model_status")
    if model_status != MODEL_STATUS:
        raise PolicyValidationError(
            f"model_status must remain {MODEL_STATUS!r}; safety validation is not implied"
        )
    policy_id = _text(raw["policy_id"], "policy_id")
    description = _text(raw["description"], "description")
    seed = _integer(raw["seed"], "seed", minimum=0, maximum=2**63 - 1)
    default_cycles = _integer(raw["default_cycles"], "default_cycles", minimum=1)
    production_hours = _number(
        raw["production_hours_per_cycle"],
        "production_hours_per_cycle",
        strictly_positive=True,
    )
    load_factor = _number(raw["load_factor"], "load_factor", strictly_positive=True)

    raw_recipes = _mapping(raw["cip_recipes"], "cip_recipes")
    if set(raw_recipes) != {"complete", "incomplete"}:
        raise PolicyValidationError(
            "cip_recipes must define exactly 'complete' and 'incomplete'"
        )
    recipes: list[CIPRecipe] = []
    canonical_recipes: dict[str, dict[str, object]] = {}
    recipe_keys = {
        "fouling_removal_fraction",
        "chemical_exposure_factor",
        "description",
    }
    for name in sorted(raw_recipes):
        item = _mapping(raw_recipes[name], f"cip_recipes.{name}")
        _exact_keys(item, recipe_keys, f"cip_recipes.{name}")
        recipe = CIPRecipe(
            name=name,
            fouling_removal_fraction=_number(
                item["fouling_removal_fraction"],
                f"cip_recipes.{name}.fouling_removal_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
            chemical_exposure_factor=_number(
                item["chemical_exposure_factor"],
                f"cip_recipes.{name}.chemical_exposure_factor",
                minimum=0.0,
            ),
            description=_text(item["description"], f"cip_recipes.{name}.description"),
        )
        recipes.append(recipe)
        canonical_recipes[name] = {
            "fouling_removal_fraction": recipe.fouling_removal_fraction,
            "chemical_exposure_factor": recipe.chemical_exposure_factor,
            "description": recipe.description,
        }
    recipe_index = {item.name: item for item in recipes}
    if (
        recipe_index["complete"].fouling_removal_fraction
        <= recipe_index["incomplete"].fouling_removal_fraction
    ):
        raise PolicyValidationError(
            "complete CIP must remove more fouling than incomplete CIP"
        )

    raw_pattern = raw["cip_pattern"]
    if not isinstance(raw_pattern, list) or not raw_pattern:
        raise PolicyValidationError("cip_pattern must be a non-empty array")
    pattern: list[str] = []
    for index, value in enumerate(raw_pattern):
        name = _text(value, f"cip_pattern[{index}]")
        if name not in recipe_index:
            raise PolicyValidationError(f"cip_pattern[{index}] refers to unknown recipe {name!r}")
        pattern.append(name)

    raw_components = raw["components"]
    if not isinstance(raw_components, list) or not raw_components:
        raise PolicyValidationError("components must be a non-empty array")
    component_keys = {
        "asset_id",
        "component_type",
        "failure_mode",
        "weibull_shape",
        "weibull_scale_h",
        "damage_rate_per_operating_h",
        "cip_damage_per_exposure",
        "fouling_rate_per_operating_h",
        "damage_limit",
        "fouling_limit",
        "priority",
    }
    components: list[ComponentSpec] = []
    asset_ids: set[str] = set()
    priorities: set[int] = set()
    canonical_components: list[dict[str, object]] = []
    for index, raw_item in enumerate(raw_components):
        context = f"components[{index}]"
        item = _mapping(raw_item, context)
        _exact_keys(item, component_keys, context)
        asset_id = _text(item["asset_id"], f"{context}.asset_id")
        priority = _integer(item["priority"], f"{context}.priority", minimum=1)
        if asset_id in asset_ids:
            raise PolicyValidationError(f"duplicate component asset_id: {asset_id}")
        if priority in priorities:
            raise PolicyValidationError(f"duplicate component priority: {priority}")
        asset_ids.add(asset_id)
        priorities.add(priority)
        component = ComponentSpec(
            asset_id=asset_id,
            component_type=_text(item["component_type"], f"{context}.component_type"),
            failure_mode=_text(item["failure_mode"], f"{context}.failure_mode"),
            weibull_shape=_number(
                item["weibull_shape"], f"{context}.weibull_shape", strictly_positive=True
            ),
            weibull_scale_h=_number(
                item["weibull_scale_h"],
                f"{context}.weibull_scale_h",
                strictly_positive=True,
            ),
            damage_rate_per_operating_h=_number(
                item["damage_rate_per_operating_h"],
                f"{context}.damage_rate_per_operating_h",
                minimum=0.0,
            ),
            cip_damage_per_exposure=_number(
                item["cip_damage_per_exposure"],
                f"{context}.cip_damage_per_exposure",
                minimum=0.0,
            ),
            fouling_rate_per_operating_h=_number(
                item["fouling_rate_per_operating_h"],
                f"{context}.fouling_rate_per_operating_h",
                minimum=0.0,
            ),
            damage_limit=_number(
                item["damage_limit"], f"{context}.damage_limit", strictly_positive=True
            ),
            fouling_limit=_number(
                item["fouling_limit"], f"{context}.fouling_limit", strictly_positive=True
            ),
            priority=priority,
        )
        components.append(component)
        canonical_components.append(
            {
                "asset_id": component.asset_id,
                "component_type": component.component_type,
                "failure_mode": component.failure_mode,
                "weibull_shape": component.weibull_shape,
                "weibull_scale_h": component.weibull_scale_h,
                "damage_rate_per_operating_h": component.damage_rate_per_operating_h,
                "cip_damage_per_exposure": component.cip_damage_per_exposure,
                "fouling_rate_per_operating_h": component.fouling_rate_per_operating_h,
                "damage_limit": component.damage_limit,
                "fouling_limit": component.fouling_limit,
                "priority": component.priority,
            }
        )
    components.sort(key=lambda item: (item.priority, item.asset_id))

    raw_actions = raw["scheduled_actions"]
    if not isinstance(raw_actions, list):
        raise PolicyValidationError("scheduled_actions must be an array")
    action_keys = {
        "asset_id",
        "every_cycles",
        "action",
        "virtual_age_factor",
        "damage_factor",
        "fouling_factor",
    }
    scheduled_actions: list[MaintenanceAction] = []
    action_assets: set[str] = set()
    canonical_actions: list[dict[str, object]] = []
    for index, raw_item in enumerate(raw_actions):
        context = f"scheduled_actions[{index}]"
        item = _mapping(raw_item, context)
        _exact_keys(item, action_keys, context)
        asset_id = _text(item["asset_id"], f"{context}.asset_id")
        if asset_id not in asset_ids:
            raise PolicyValidationError(f"{context} refers to unknown asset {asset_id!r}")
        if asset_id in action_assets:
            raise PolicyValidationError(
                f"multiple scheduled actions for asset {asset_id!r} are ambiguous"
            )
        action_assets.add(asset_id)
        raw_name = _text(item["action"], f"{context}.action")
        normalized = _normal_action_name(raw_name)
        action = MaintenanceAction(
            asset_id=asset_id,
            every_cycles=_integer(
                item["every_cycles"], f"{context}.every_cycles", minimum=1
            ),
            action=normalized,
            virtual_age_factor=_number(
                item["virtual_age_factor"],
                f"{context}.virtual_age_factor",
                minimum=0.0,
                maximum=1.0,
            ),
            damage_factor=_number(
                item["damage_factor"],
                f"{context}.damage_factor",
                minimum=0.0,
                maximum=1.0,
            ),
            fouling_factor=_number(
                item["fouling_factor"],
                f"{context}.fouling_factor",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        scheduled_actions.append(action)
        canonical_actions.append(
            {
                "asset_id": asset_id,
                "every_cycles": action.every_cycles,
                "action": raw_name,
                "virtual_age_factor": action.virtual_age_factor,
                "damage_factor": action.damage_factor,
                "fouling_factor": action.fouling_factor,
            }
        )

    corrective_raw = _mapping(raw["corrective_action"], "corrective_action")
    corrective_keys = action_keys - {"asset_id", "every_cycles"}
    _exact_keys(corrective_raw, corrective_keys, "corrective_action")
    corrective_name = _normal_action_name(
        _text(corrective_raw["action"], "corrective_action.action")
    )
    corrective = MaintenanceAction(
        asset_id=None,
        every_cycles=None,
        action=corrective_name,
        virtual_age_factor=_number(
            corrective_raw["virtual_age_factor"],
            "corrective_action.virtual_age_factor",
            minimum=0.0,
            maximum=1.0,
        ),
        damage_factor=_number(
            corrective_raw["damage_factor"],
            "corrective_action.damage_factor",
            minimum=0.0,
            maximum=1.0,
        ),
        fouling_factor=_number(
            corrective_raw["fouling_factor"],
            "corrective_action.fouling_factor",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if corrective.action != "replacement" or any(
        value != 0.0
        for value in (
            corrective.virtual_age_factor,
            corrective.damage_factor,
            corrective.fouling_factor,
        )
    ):
        raise PolicyValidationError(
            "corrective_action must be a zero-factor replacement"
        )

    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PolicyValidationError("sources must be a non-empty array")
    sources: list[dict[str, str]] = []
    source_ids: set[str] = set()
    for index, raw_item in enumerate(raw_sources):
        context = f"sources[{index}]"
        item = _mapping(raw_item, context)
        _exact_keys(item, {"source_id", "url", "use"}, context)
        source_id = _text(item["source_id"], f"{context}.source_id")
        url = _text(item["url"], f"{context}.url")
        use = _text(item["use"], f"{context}.use")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise PolicyValidationError(f"{context}.url must be an HTTP(S) URL")
        if source_id in source_ids:
            raise PolicyValidationError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        sources.append({"source_id": source_id, "url": url, "use": use})

    canonical = {
        "schema_version": schema_version,
        "policy_id": policy_id,
        "description": description,
        "model_status": model_status,
        "seed": seed,
        "default_cycles": default_cycles,
        "production_hours_per_cycle": production_hours,
        "load_factor": load_factor,
        "cip_pattern": pattern,
        "cip_recipes": canonical_recipes,
        "components": canonical_components,
        "scheduled_actions": canonical_actions,
        "corrective_action": {
            "action": str(corrective_raw["action"]),
            "virtual_age_factor": corrective.virtual_age_factor,
            "damage_factor": corrective.damage_factor,
            "fouling_factor": corrective.fouling_factor,
        },
        "sources": sources,
    }
    return MaintenancePolicy(
        schema_version=schema_version,
        policy_id=policy_id,
        description=description,
        model_status=model_status,
        seed=seed,
        default_cycles=default_cycles,
        production_hours_per_cycle=production_hours,
        load_factor=load_factor,
        cip_pattern=tuple(pattern),
        cip_recipes=tuple(recipes),
        components=tuple(components),
        scheduled_actions=tuple(scheduled_actions),
        corrective_action=corrective,
        sources=tuple(copy.deepcopy(sources)),
        canonical_payload=canonical,
    )


def load_maintenance_policy(path: Path) -> MaintenancePolicy:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyValidationError(f"cannot load maintenance policy {path}: {exc}") from exc
    return validate_policy(_mapping(payload, "policy"))


def canonical_policy_hash(policy: MaintenancePolicy) -> str:
    encoded = json.dumps(
        policy.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weibull_hazard(spec: ComponentSpec, virtual_age_h: float) -> float:
    return (max(0.0, virtual_age_h) / spec.weibull_scale_h) ** spec.weibull_shape


def _draw_unit_exponential(rng: random.Random) -> float:
    return -math.log(max(rng.random(), 2.0**-53))


def conditional_median_next_event_rul(
    spec: ComponentSpec,
    *,
    virtual_age_h: float,
    irreversible_damage: float,
    reversible_fouling: float,
    load_factor: float,
) -> dict[str, float | str | None]:
    """Return production-hours-to-risk medians conditional on current state.

    The stochastic Weibull risk uses its conditional median.  Damage and
    fouling thresholds are deterministic competing risks.  Planned future CIP
    and service actions are intentionally not anticipated by this diagnostic.
    """

    for name, value in {
        "virtual_age_h": virtual_age_h,
        "irreversible_damage": irreversible_damage,
        "reversible_fouling": reversible_fouling,
        "load_factor": load_factor,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if load_factor <= 0.0:
        raise ValueError("load_factor must be positive")

    current_hazard = _weibull_hazard(spec, virtual_age_h)
    median_target = current_hazard + math.log(2.0)
    median_age = spec.weibull_scale_h * median_target ** (1.0 / spec.weibull_shape)
    weibull_rul = max(0.0, median_age - virtual_age_h) / load_factor

    damage_rate = spec.damage_rate_per_operating_h * load_factor
    if irreversible_damage >= spec.damage_limit:
        damage_rul: float | None = 0.0
    elif damage_rate > 0.0:
        damage_rul = (spec.damage_limit - irreversible_damage) / damage_rate
    else:
        damage_rul = None

    fouling_rate = spec.fouling_rate_per_operating_h * load_factor
    if reversible_fouling >= spec.fouling_limit:
        fouling_rul: float | None = 0.0
    elif fouling_rate > 0.0:
        fouling_rul = (spec.fouling_limit - reversible_fouling) / fouling_rate
    else:
        fouling_rul = None

    candidates = [(weibull_rul, "WEIBULL_FAILURE")]
    if damage_rul is not None:
        candidates.append((damage_rul, "DAMAGE_LIMIT"))
    if fouling_rul is not None:
        candidates.append((fouling_rul, "FOULING_LIMIT"))
    cause_rank = {name: index for index, name in enumerate(FAILURE_CAUSE_PRIORITY)}
    competing_rul, cause = min(
        candidates,
        key=lambda item: (item[0], cause_rank[item[1]]),
    )
    return {
        "weibull_median_rul_h": weibull_rul,
        "damage_limit_rul_h": damage_rul,
        "fouling_limit_rul_h": fouling_rul,
        "conditional_median_next_event_rul_h": competing_rul,
        "median_competing_cause": cause,
    }


class _LifecycleEngine:
    def __init__(self, policy: MaintenancePolicy, *, cycles: int, seed: int) -> None:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be an integer in 0..2^63-1")
        self.policy = policy
        self.cycles = cycles
        self.seed = seed
        self.policy_hash = canonical_policy_hash(policy)
        run_payload = {
            "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
            "policy_hash": self.policy_hash,
            "cycles": cycles,
            "seed": seed,
        }
        run_digest = hashlib.sha256(
            json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.run_id = f"MC-RUL-{run_digest[:20]}"
        self.rng = random.Random(seed)
        self.states: dict[str, ComponentState] = {}
        for spec in policy.components:
            self.states[spec.asset_id] = ComponentState(
                spec=spec,
                virtual_age_h=0.0,
                irreversible_damage=0.0,
                reversible_fouling=0.0,
                weibull_remaining_hazard=_draw_unit_exponential(self.rng),
                renewal_count=0,
                interval_index=0,
                interval_start_time_h=0.0,
                interval_start_cycle=0,
                interval_start_action="initial",
                interval_start_virtual_age_h=0.0,
                interval_start_damage=0.0,
                interval_start_fouling=0.0,
            )
        self.time_h = 0.0
        self.trace: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.intervals: list[dict[str, object]] = []
        self.event_sequence = 0

    def _event_id(self) -> str:
        self.event_sequence += 1
        return f"EV-{self.event_sequence:07d}"

    def _ordered_states(self) -> list[ComponentState]:
        return sorted(
            self.states.values(), key=lambda state: (state.spec.priority, state.spec.asset_id)
        )

    def _candidate(self, state: ComponentState) -> tuple[float, str]:
        spec = state.spec
        load = self.policy.load_factor
        target_hazard = _weibull_hazard(spec, state.virtual_age_h) + max(
            0.0, state.weibull_remaining_hazard
        )
        failure_age = spec.weibull_scale_h * target_hazard ** (1.0 / spec.weibull_shape)
        weibull_time = max(0.0, failure_age - state.virtual_age_h) / load
        candidates: list[tuple[float, str]] = [(weibull_time, "WEIBULL_FAILURE")]
        damage_rate = spec.damage_rate_per_operating_h * load
        if state.irreversible_damage >= spec.damage_limit:
            candidates.append((0.0, "DAMAGE_LIMIT"))
        elif damage_rate > 0.0:
            candidates.append(
                (
                    (spec.damage_limit - state.irreversible_damage) / damage_rate,
                    "DAMAGE_LIMIT",
                )
            )
        fouling_rate = spec.fouling_rate_per_operating_h * load
        if state.reversible_fouling >= spec.fouling_limit:
            candidates.append((0.0, "FOULING_LIMIT"))
        elif fouling_rate > 0.0:
            candidates.append(
                (
                    (spec.fouling_limit - state.reversible_fouling) / fouling_rate,
                    "FOULING_LIMIT",
                )
            )
        rank = {name: index for index, name in enumerate(FAILURE_CAUSE_PRIORITY)}
        return min(candidates, key=lambda item: (item[0], rank[item[1]]))

    def _advance_all(self, duration_h: float) -> None:
        if duration_h < 0.0 or not math.isfinite(duration_h):
            raise RuntimeError("invalid lifecycle time increment")
        load = self.policy.load_factor
        for state in self._ordered_states():
            before_hazard = _weibull_hazard(state.spec, state.virtual_age_h)
            state.virtual_age_h += duration_h * load
            after_hazard = _weibull_hazard(state.spec, state.virtual_age_h)
            state.weibull_remaining_hazard = max(
                0.0,
                state.weibull_remaining_hazard - (after_hazard - before_hazard),
            )
            state.irreversible_damage += (
                state.spec.damage_rate_per_operating_h * duration_h * load
            )
            state.reversible_fouling += (
                state.spec.fouling_rate_per_operating_h * duration_h * load
            )
        self.time_h += duration_h

    def _close_interval(
        self,
        state: ComponentState,
        *,
        cycle: int,
        terminal_event_type: str,
        failure_cause: str,
        event_observed: bool,
    ) -> None:
        duration = max(0.0, self.time_h - state.interval_start_time_h)
        self.intervals.append(
            {
                "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
                "run_id": self.run_id,
                "service_interval_id": state.service_interval_id,
                "asset_id": state.spec.asset_id,
                "component_type": state.spec.component_type,
                "interval_index": state.interval_index,
                "generation": state.renewal_count,
                "start_time_h": state.interval_start_time_h,
                "end_time_h": self.time_h,
                "duration_h": duration,
                "start_cycle": state.interval_start_cycle,
                "end_cycle": cycle,
                "start_action": state.interval_start_action,
                "terminal_event_type": terminal_event_type,
                "event_observed": int(event_observed),
                "failure_cause": failure_cause,
                "exact_rul_h": duration if event_observed else None,
                "rul_lower_bound_h": None if event_observed else duration,
                "start_virtual_age_h": state.interval_start_virtual_age_h,
                "end_virtual_age_h": state.virtual_age_h,
                "start_damage": state.interval_start_damage,
                "end_damage": state.irreversible_damage,
                "start_fouling": state.interval_start_fouling,
                "end_fouling": state.reversible_fouling,
                "is_last_interval": 0,
            }
        )

    def _open_interval(
        self, state: ComponentState, *, cycle: int, start_action: str
    ) -> None:
        state.interval_index += 1
        state.interval_start_time_h = self.time_h
        state.interval_start_cycle = cycle
        state.interval_start_action = start_action
        state.interval_start_virtual_age_h = state.virtual_age_h
        state.interval_start_damage = state.irreversible_damage
        state.interval_start_fouling = state.reversible_fouling

    def _apply_action(
        self,
        state: ComponentState,
        action: MaintenanceAction,
    ) -> tuple[float, float, float]:
        before = (
            state.virtual_age_h,
            state.irreversible_damage,
            state.reversible_fouling,
        )
        # Kijima-II virtual-age update: V_after = q * V_before.
        state.virtual_age_h = action.virtual_age_factor * state.virtual_age_h
        state.irreversible_damage = action.damage_factor * state.irreversible_damage
        state.reversible_fouling = action.fouling_factor * state.reversible_fouling
        state.weibull_remaining_hazard = _draw_unit_exponential(self.rng)
        if action.action == "replacement":
            state.renewal_count += 1
        return before

    def _record_event(
        self,
        state: ComponentState,
        *,
        cycle: int,
        event_type: str,
        trigger_stage: str,
        failure_cause: str,
        maintenance_action: str,
        cip_kind: str,
        service_interval_id: str,
        event_observed: int,
        before: tuple[float, float, float],
        factors: tuple[float, float, float],
    ) -> None:
        self.events.append(
            {
                "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
                "run_id": self.run_id,
                "event_id": self._event_id(),
                "time_h": self.time_h,
                "cycle": cycle,
                "asset_id": state.spec.asset_id,
                "component_type": state.spec.component_type,
                "event_type": event_type,
                "trigger_stage": trigger_stage,
                "failure_cause": failure_cause,
                "maintenance_action": maintenance_action,
                "cip_kind": cip_kind,
                "service_interval_id": service_interval_id,
                "event_observed": event_observed,
                "virtual_age_before_h": before[0],
                "virtual_age_after_h": state.virtual_age_h,
                "damage_before": before[1],
                "damage_after": state.irreversible_damage,
                "fouling_before": before[2],
                "fouling_after": state.reversible_fouling,
                "virtual_age_factor_q": factors[0],
                "damage_factor": factors[1],
                "fouling_factor": factors[2],
            }
        )

    def _fail_and_replace(
        self,
        state: ComponentState,
        *,
        cycle: int,
        cause: str,
        trigger_stage: str,
    ) -> None:
        interval_id = state.service_interval_id
        self._close_interval(
            state,
            cycle=cycle,
            terminal_event_type="failure",
            failure_cause=cause,
            event_observed=True,
        )
        before = self._apply_action(state, self.policy.corrective_action)
        factors = (
            self.policy.corrective_action.virtual_age_factor,
            self.policy.corrective_action.damage_factor,
            self.policy.corrective_action.fouling_factor,
        )
        self._record_event(
            state,
            cycle=cycle,
            event_type="failure",
            trigger_stage=trigger_stage,
            failure_cause=cause,
            maintenance_action="replacement",
            cip_kind="",
            service_interval_id=interval_id,
            event_observed=1,
            before=before,
            factors=factors,
        )
        self._open_interval(state, cycle=cycle, start_action="replacement")

    def _run_production(self, cycle: int) -> dict[str, int]:
        remaining = self.policy.production_hours_per_cycle
        failure_counts = {state.spec.asset_id: 0 for state in self._ordered_states()}
        event_guard = 0
        while remaining > 1.0e-12:
            candidates: list[tuple[float, int, int, str, str]] = []
            cause_rank = {
                name: index for index, name in enumerate(FAILURE_CAUSE_PRIORITY)
            }
            for state in self._ordered_states():
                delay, cause = self._candidate(state)
                candidates.append(
                    (delay, state.spec.priority, cause_rank[cause], state.spec.asset_id, cause)
                )
            delay, _, _, asset_id, cause = min(candidates)
            if delay > remaining + 1.0e-12:
                self._advance_all(remaining)
                remaining = 0.0
                break
            elapsed = max(0.0, min(delay, remaining))
            self._advance_all(elapsed)
            remaining -= elapsed
            self._fail_and_replace(
                self.states[asset_id],
                cycle=cycle,
                cause=cause,
                trigger_stage="production",
            )
            failure_counts[asset_id] += 1
            event_guard += 1
            if event_guard > 100_000:
                raise RuntimeError("excessive recurrent failures; check lifecycle policy")
        return failure_counts

    def _apply_cip(
        self, state: ComponentState, *, cycle: int, recipe: CIPRecipe
    ) -> tuple[tuple[float, float, float], bool]:
        before = (
            state.virtual_age_h,
            state.irreversible_damage,
            state.reversible_fouling,
        )
        state.irreversible_damage += (
            state.spec.cip_damage_per_exposure * recipe.chemical_exposure_factor
        )
        state.reversible_fouling *= 1.0 - recipe.fouling_removal_fraction
        self._record_event(
            state,
            cycle=cycle,
            event_type="cip",
            trigger_stage="cip",
            failure_cause="",
            maintenance_action=f"cip_{recipe.name}",
            cip_kind=recipe.name,
            service_interval_id=state.service_interval_id,
            event_observed=0,
            before=before,
            factors=(1.0, 1.0, 1.0 - recipe.fouling_removal_fraction),
        )
        failed = state.irreversible_damage >= state.spec.damage_limit
        if failed:
            self._fail_and_replace(
                state,
                cycle=cycle,
                cause="DAMAGE_LIMIT",
                trigger_stage="cip",
            )
        return before, failed

    def _scheduled_service(
        self, state: ComponentState, *, cycle: int, action: MaintenanceAction
    ) -> None:
        interval_id = state.service_interval_id
        self._close_interval(
            state,
            cycle=cycle,
            terminal_event_type="scheduled_maintenance",
            failure_cause="",
            event_observed=False,
        )
        before = self._apply_action(state, action)
        factors = (
            action.virtual_age_factor,
            action.damage_factor,
            action.fouling_factor,
        )
        self._record_event(
            state,
            cycle=cycle,
            event_type="scheduled_maintenance",
            trigger_stage="cycle_end",
            failure_cause="",
            maintenance_action=action.action,
            cip_kind="",
            service_interval_id=interval_id,
            event_observed=0,
            before=before,
            factors=factors,
        )
        self._open_interval(state, cycle=cycle, start_action=action.action)

    def _trace_row(
        self,
        state: ComponentState,
        *,
        cycle: int,
        cip_kind: str,
        scheduled_action: str,
        failures_in_cycle: int,
        before_production: tuple[float, float, float],
        before_cip: tuple[float, float, float],
        after_cip: tuple[float, float],
    ) -> dict[str, object]:
        median = conditional_median_next_event_rul(
            state.spec,
            virtual_age_h=state.virtual_age_h,
            irreversible_damage=state.irreversible_damage,
            reversible_fouling=state.reversible_fouling,
            load_factor=self.policy.load_factor,
        )
        return {
            "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
            "run_id": self.run_id,
            "policy_id": self.policy.policy_id,
            "policy_hash": self.policy_hash,
            "seed": self.seed,
            "cycle": cycle,
            "cycle_end_time_h": self.time_h,
            "asset_id": state.spec.asset_id,
            "component_type": state.spec.component_type,
            "failure_mode": state.spec.failure_mode,
            "cip_kind": cip_kind,
            "scheduled_action": scheduled_action,
            "failures_in_cycle": failures_in_cycle,
            "renewal_count": state.renewal_count,
            "service_interval_id": state.service_interval_id,
            "virtual_age_before_production_h": before_production[0],
            "damage_before_production": before_production[1],
            "fouling_before_production": before_production[2],
            "virtual_age_before_cip_h": before_cip[0],
            "damage_before_cip": before_cip[1],
            "fouling_before_cip": before_cip[2],
            "damage_after_cip": after_cip[0],
            "fouling_after_cip": after_cip[1],
            "virtual_age_h": state.virtual_age_h,
            "irreversible_damage": state.irreversible_damage,
            "reversible_fouling": state.reversible_fouling,
            **median,
        }

    def run(self) -> LifecycleResult:
        for state in self._ordered_states():
            zero = (0.0, 0.0, 0.0)
            self.trace.append(
                self._trace_row(
                    state,
                    cycle=0,
                    cip_kind="",
                    scheduled_action="none",
                    failures_in_cycle=0,
                    before_production=zero,
                    before_cip=zero,
                    after_cip=(0.0, 0.0),
                )
            )

        action_index = self.policy.action_by_asset
        recipes = self.policy.recipe_by_name
        for cycle in range(1, self.cycles + 1):
            before_production = {
                state.spec.asset_id: (
                    state.virtual_age_h,
                    state.irreversible_damage,
                    state.reversible_fouling,
                )
                for state in self._ordered_states()
            }
            failures = self._run_production(cycle)
            cip_kind = self.policy.cip_pattern[(cycle - 1) % len(self.policy.cip_pattern)]
            recipe = recipes[cip_kind]
            before_cip: dict[str, tuple[float, float, float]] = {}
            after_cip: dict[str, tuple[float, float]] = {}
            cip_corrective: set[str] = set()
            for state in self._ordered_states():
                before = (
                    state.virtual_age_h,
                    state.irreversible_damage,
                    state.reversible_fouling,
                )
                before_cip[state.spec.asset_id] = before
                _, failed = self._apply_cip(state, cycle=cycle, recipe=recipe)
                after_cip[state.spec.asset_id] = (
                    before[1]
                    + state.spec.cip_damage_per_exposure * recipe.chemical_exposure_factor,
                    before[2] * (1.0 - recipe.fouling_removal_fraction),
                )
                if failed:
                    cip_corrective.add(state.spec.asset_id)
                    failures[state.spec.asset_id] += 1

            scheduled_names: dict[str, str] = {
                state.spec.asset_id: "none" for state in self._ordered_states()
            }
            for state in self._ordered_states():
                action = action_index.get(state.spec.asset_id)
                if (
                    action is not None
                    and action.every_cycles is not None
                    and cycle % action.every_cycles == 0
                ):
                    if state.spec.asset_id in cip_corrective:
                        scheduled_names[state.spec.asset_id] = "skipped_after_replacement"
                    else:
                        self._scheduled_service(state, cycle=cycle, action=action)
                        scheduled_names[state.spec.asset_id] = action.action

            for state in self._ordered_states():
                asset_id = state.spec.asset_id
                self.trace.append(
                    self._trace_row(
                        state,
                        cycle=cycle,
                        cip_kind=cip_kind,
                        scheduled_action=scheduled_names[asset_id],
                        failures_in_cycle=failures[asset_id],
                        before_production=before_production[asset_id],
                        before_cip=before_cip[asset_id],
                        after_cip=after_cip[asset_id],
                    )
                )

        for state in self._ordered_states():
            self._close_interval(
                state,
                cycle=self.cycles,
                terminal_event_type="administrative_censor",
                failure_cause="",
                event_observed=False,
            )
        last_by_asset: dict[str, dict[str, object]] = {}
        for interval in self.intervals:
            last_by_asset[str(interval["asset_id"])] = interval
        for interval in last_by_asset.values():
            interval["is_last_interval"] = 1

        failures = [item for item in self.events if item["event_type"] == "failure"]
        maintenance = [
            item for item in self.events if item["event_type"] == "scheduled_maintenance"
        ]
        cips = [item for item in self.events if item["event_type"] == "cip"]
        causes: dict[str, int] = {}
        for event in failures:
            cause = str(event["failure_cause"])
            causes[cause] = causes.get(cause, 0) + 1
        components: list[dict[str, object]] = []
        for state in self._ordered_states():
            final_trace = next(
                row
                for row in reversed(self.trace)
                if row["asset_id"] == state.spec.asset_id
            )
            asset_intervals = [
                row for row in self.intervals if row["asset_id"] == state.spec.asset_id
            ]
            components.append(
                {
                    "asset_id": state.spec.asset_id,
                    "component_type": state.spec.component_type,
                    "failure_mode": state.spec.failure_mode,
                    "failure_count": sum(
                        event["asset_id"] == state.spec.asset_id for event in failures
                    ),
                    "scheduled_maintenance_count": sum(
                        event["asset_id"] == state.spec.asset_id for event in maintenance
                    ),
                    "service_interval_count": len(asset_intervals),
                    "renewal_count": state.renewal_count,
                    "final_virtual_age_h": state.virtual_age_h,
                    "final_irreversible_damage": state.irreversible_damage,
                    "final_reversible_fouling": state.reversible_fouling,
                    "conditional_median_next_event_rul_h": final_trace[
                        "conditional_median_next_event_rul_h"
                    ],
                    "median_competing_cause": final_trace["median_competing_cause"],
                    "last_interval_right_censored": bool(
                        asset_intervals
                        and asset_intervals[-1]["event_observed"] == 0
                        and asset_intervals[-1]["exact_rul_h"] is None
                    ),
                }
            )
        summary: dict[str, object] = {
            "model_status": MODEL_STATUS,
            "lifecycle_model_version": LIFECYCLE_MODEL_VERSION,
            "policy_schema_version": self.policy.schema_version,
            "policy_id": self.policy.policy_id,
            "policy_hash": self.policy_hash,
            "run_id": self.run_id,
            "seed": self.seed,
            "cycles": self.cycles,
            "production_hours_per_cycle": self.policy.production_hours_per_cycle,
            "total_operating_hours": self.cycles * self.policy.production_hours_per_cycle,
            "counts": {
                "trace_rows": len(self.trace),
                "events": len(self.events),
                "cip_events": len(cips),
                "scheduled_maintenance_events": len(maintenance),
                "failure_events": len(failures),
                "service_intervals": len(self.intervals),
            },
            "failure_counts_by_cause": dict(sorted(causes.items())),
            "cip_counts": {
                name: sum(
                    1
                    for cycle in range(1, self.cycles + 1)
                    if self.policy.cip_pattern[(cycle - 1) % len(self.policy.cip_pattern)]
                    == name
                )
                for name in sorted(recipes)
            },
            "all_last_intervals_right_censored": all(
                component["last_interval_right_censored"] for component in components
            ),
            "components": components,
            "sources": [dict(item) for item in self.policy.sources],
            "limitations": [
                "Synthetic public-reference lifecycle surrogate; parameters are not fitted to plant maintenance records.",
                "Conditional median RUL assumes continued production load and does not anticipate future scheduled CIP or maintenance.",
                "Exact RUL is null for right-censored intervals; the observed survival duration is stored only as a lower bound.",
            ],
        }
        return LifecycleResult(
            run_id=self.run_id,
            policy_hash=self.policy_hash,
            seed=self.seed,
            cycles=self.cycles,
            trace=self.trace,
            events=self.events,
            service_intervals=self.intervals,
            summary=summary,
        )


def simulate_lifecycle(
    policy: MaintenancePolicy,
    *,
    cycles: int | None = None,
    seed: int | None = None,
) -> LifecycleResult:
    """Run a deterministic recurrent-event lifecycle simulation."""

    resolved_cycles = policy.default_cycles if cycles is None else cycles
    resolved_seed = policy.seed if seed is None else seed
    return _LifecycleEngine(
        policy,
        cycles=resolved_cycles,
        seed=resolved_seed,
    ).run()


def rows_have_exact_fields(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bool:
    expected = list(fields)
    return all(list(row) == expected for row in rows)
