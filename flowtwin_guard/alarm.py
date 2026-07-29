"""Validation-only operational alarm policy for FlowTwin experiments.

This module deliberately separates row-level anomaly scores from an alarm
state.  The state machine is causal, resets between episodes, and applies
threshold hysteresis, assertion/clear persistence, and a post-clear cooldown.

Policy selection is restricted to ``validation_id`` rows.  A candidate is
feasible only when *every* validation profile has measurable truth-negative
exposure and its false-alarm-onset rate satisfies the configured limit.  The
result is an auditable research operating point, not a product-safety, HACCP,
or release decision.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import event_detection_metrics, false_alarm_metrics


ALARM_POLICY_VERSION = "0.1.0"
_TIME_ATOL = 1e-12
_VALIDATION_SPLIT = "validation_id"


def _finite_float(name: str, value: object, *, non_negative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _one_dimensional(name: str, values: Sequence[object] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array


def _float_vector(
    name: str,
    values: Sequence[float] | np.ndarray,
    *,
    length: int | None = None,
    strictly_positive: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and int(array.size) != length:
        raise ValueError(f"{name} must contain {length} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if strictly_positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def _json_scalar(name: str, value: object) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)) or value is None:
        raise ValueError(f"{name} must contain JSON scalar identifiers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must not contain non-finite identifiers")
    return value


def _identifier_vector(
    name: str,
    values: Sequence[object] | np.ndarray,
    *,
    length: int,
) -> list[str | int | float | bool]:
    array = _one_dimensional(name, values)
    if int(array.size) != length:
        raise ValueError(f"{name} must contain {length} values")
    return [_json_scalar(name, value) for value in array.tolist()]


def _binary_vector(
    name: str,
    values: Sequence[object] | np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    array = _one_dimensional(name, values)
    if int(array.size) != length:
        raise ValueError(f"{name} must contain {length} values")
    if array.dtype.kind in "fc":
        numeric = np.asarray(array, dtype=np.float64)
        valid = np.isfinite(numeric).all() and np.isin(numeric, (0.0, 1.0)).all()
    elif array.dtype.kind in "iub":
        valid = np.isin(array, (0, 1, False, True)).all()
    else:
        valid = False
    if not valid:
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(np.bool_, copy=False)


def _stable_json_unique(values: Sequence[str | int | float | bool]) -> list[Any]:
    # JSON text includes the scalar type, avoiding collisions such as 1 and "1".
    keyed = {json.dumps(value, ensure_ascii=False): value for value in values}
    return [keyed[key] for key in sorted(keyed)]


@dataclass(frozen=True)
class AlarmPolicyConfig:
    """Parameters for the causal alarm state machine.

    ``min_on_duration_s`` is the continuous duration at or above the assertion
    threshold required before activation. ``min_off_duration_s`` is the
    continuous duration at or below the clear threshold required before
    clearing. ``cooldown_s`` inhibits reactivation after a clear.
    """

    on_threshold: float
    off_threshold: float
    min_on_duration_s: float = 0.0
    min_off_duration_s: float = 0.0
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        on_threshold = _finite_float("on_threshold", self.on_threshold)
        off_threshold = _finite_float("off_threshold", self.off_threshold)
        min_on = _finite_float(
            "min_on_duration_s", self.min_on_duration_s, non_negative=True
        )
        min_off = _finite_float(
            "min_off_duration_s", self.min_off_duration_s, non_negative=True
        )
        cooldown = _finite_float("cooldown_s", self.cooldown_s, non_negative=True)
        if on_threshold < off_threshold:
            raise ValueError("on_threshold must be greater than or equal to off_threshold")
        object.__setattr__(self, "on_threshold", on_threshold)
        object.__setattr__(self, "off_threshold", off_threshold)
        object.__setattr__(self, "min_on_duration_s", min_on)
        object.__setattr__(self, "min_off_duration_s", min_off)
        object.__setattr__(self, "cooldown_s", cooldown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": ALARM_POLICY_VERSION,
            "on_threshold": self.on_threshold,
            "off_threshold": self.off_threshold,
            "min_on_duration_s": self.min_on_duration_s,
            "min_off_duration_s": self.min_off_duration_s,
            "cooldown_s": self.cooldown_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AlarmPolicyConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("alarm policy config must be a mapping")
        version = payload.get("version", ALARM_POLICY_VERSION)
        if version != ALARM_POLICY_VERSION:
            raise ValueError(
                f"unsupported alarm policy config version {version!r}; "
                f"expected {ALARM_POLICY_VERSION!r}"
            )
        required = {
            "on_threshold",
            "off_threshold",
            "min_on_duration_s",
            "min_off_duration_s",
            "cooldown_s",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"alarm policy config is missing fields: {missing}")
        unexpected = sorted(set(payload) - required - {"version"})
        if unexpected:
            raise ValueError(f"alarm policy config has unexpected fields: {unexpected}")
        return cls(**{name: payload[name] for name in required})  # type: ignore[arg-type]


@dataclass
class _AlarmState:
    active: bool = False
    on_elapsed_s: float = 0.0
    off_elapsed_s: float = 0.0
    cooldown_remaining_s: float = 0.0


def apply_alarm_policy(
    scores: Sequence[float] | np.ndarray,
    step_dt_s: Sequence[float] | np.ndarray,
    episode_ids: Sequence[object] | np.ndarray,
    config: AlarmPolicyConfig | Mapping[str, object],
) -> np.ndarray:
    """Apply a causal policy and return a Boolean alarm vector.

    Rows are consumed in caller-provided order. Each episode owns independent
    state, so interleaved episodes cannot leak persistence or cooldown into one
    another. A threshold duration includes the current row's ``step_dt_s`` and
    the returned state is the state after consuming that row.
    """

    policy = (
        config
        if isinstance(config, AlarmPolicyConfig)
        else AlarmPolicyConfig.from_dict(config)
    )
    score = _float_vector("scores", scores)
    n = int(score.size)
    durations = _float_vector(
        "step_dt_s", step_dt_s, length=n, strictly_positive=True
    )
    episodes = _identifier_vector("episode_ids", episode_ids, length=n)
    alarms = np.zeros(n, dtype=np.bool_)
    states: dict[str, _AlarmState] = {}

    for index, (value, duration, episode_id) in enumerate(
        zip(score, durations, episodes, strict=True)
    ):
        episode_key = json.dumps(episode_id, ensure_ascii=False)
        state = states.setdefault(episode_key, _AlarmState())
        dt_s = float(duration)
        scalar_score = float(value)

        if state.active:
            if scalar_score <= policy.off_threshold:
                state.off_elapsed_s += dt_s
                if (
                    policy.min_off_duration_s == 0.0
                    or state.off_elapsed_s + _TIME_ATOL
                    >= policy.min_off_duration_s
                ):
                    state.active = False
                    state.on_elapsed_s = 0.0
                    state.off_elapsed_s = 0.0
                    state.cooldown_remaining_s = policy.cooldown_s
            else:
                state.off_elapsed_s = 0.0
        else:
            available_dt_s = dt_s
            if state.cooldown_remaining_s > 0.0:
                inhibited_s = min(available_dt_s, state.cooldown_remaining_s)
                state.cooldown_remaining_s = max(
                    0.0, state.cooldown_remaining_s - inhibited_s
                )
                available_dt_s -= inhibited_s
                state.on_elapsed_s = 0.0

            if available_dt_s > _TIME_ATOL:
                if scalar_score >= policy.on_threshold:
                    state.on_elapsed_s += available_dt_s
                    if (
                        policy.min_on_duration_s == 0.0
                        or state.on_elapsed_s + _TIME_ATOL
                        >= policy.min_on_duration_s
                    ):
                        state.active = True
                        state.on_elapsed_s = 0.0
                        state.off_elapsed_s = 0.0
                else:
                    state.on_elapsed_s = 0.0

        alarms[index] = state.active

    return alarms


def _grid(name: str, values: Sequence[float], *, non_negative: bool) -> list[float]:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    result = sorted(
        {
            _finite_float(name, value, non_negative=non_negative)
            for value in values
        }
    )
    return result


def _candidate_configs(
    *,
    on_thresholds: Sequence[float],
    off_thresholds: Sequence[float],
    min_on_durations_s: Sequence[float],
    min_off_durations_s: Sequence[float],
    cooldowns_s: Sequence[float],
) -> list[AlarmPolicyConfig]:
    ons = _grid("on_thresholds", on_thresholds, non_negative=False)
    offs = _grid("off_thresholds", off_thresholds, non_negative=False)
    min_ons = _grid(
        "min_on_durations_s", min_on_durations_s, non_negative=True
    )
    min_offs = _grid(
        "min_off_durations_s", min_off_durations_s, non_negative=True
    )
    cooldowns = _grid("cooldowns_s", cooldowns_s, non_negative=True)
    result = [
        AlarmPolicyConfig(on, off, min_on, min_off, cooldown)
        for on in ons
        for off in offs
        for min_on in min_ons
        for min_off in min_offs
        for cooldown in cooldowns
        if on >= off
    ]
    if not result:
        raise ValueError("candidate grid has no on_threshold >= off_threshold pair")
    return result


def _profile_false_alarm_audit(
    *,
    truth_active: np.ndarray,
    alarm_active: np.ndarray,
    time_s: np.ndarray,
    step_dt_s: np.ndarray,
    episode_ids: list[str | int | float | bool],
    profile_ids: list[str | int | float | bool],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rates: list[float] = []
    missing_negative_exposure: list[Any] = []
    profile_order = _stable_json_unique(profile_ids)
    profile_array = np.asarray(
        [json.dumps(value, ensure_ascii=False) for value in profile_ids], dtype=object
    )
    episode_array = np.asarray(episode_ids, dtype=object)
    for profile_id in profile_order:
        key = json.dumps(profile_id, ensure_ascii=False)
        mask = profile_array == key
        metrics = false_alarm_metrics(
            truth_active[mask],
            alarm_active[mask],
            time_s[mask],
            step_dt_s[mask],
            episode_ids=episode_array[mask],
        )
        rate = metrics["false_alarm_onsets_per_negative_hour"]
        if rate is None:
            missing_negative_exposure.append(profile_id)
        else:
            rates.append(float(rate))
        rows.append(
            {
                "profile_id": profile_id,
                "negative_duration_s": metrics["negative_duration_s"],
                "false_alarm_onsets": metrics["false_alarm_onsets"],
                "false_alarm_onsets_per_negative_hour": rate,
            }
        )

    pooled = false_alarm_metrics(
        truth_active,
        alarm_active,
        time_s,
        step_dt_s,
        episode_ids=episode_ids,
    )
    return {
        "aggregation": "maximum per-profile onset rate",
        "profiles": rows,
        "profiles_without_negative_exposure": missing_negative_exposure,
        "all_profiles_have_negative_exposure": not missing_negative_exposure,
        "maximum_profile_false_alarm_onsets_per_negative_hour": (
            max(rates) if rates and not missing_negative_exposure else None
        ),
        "mean_profile_false_alarm_onsets_per_negative_hour": (
            float(np.mean(rates)) if rates and not missing_negative_exposure else None
        ),
        "pooled_false_alarm_onsets_per_negative_hour": pooled[
            "false_alarm_onsets_per_negative_hour"
        ],
        "pooled_negative_duration_s": pooled["negative_duration_s"],
        "pooled_false_alarm_onsets": pooled["false_alarm_onsets"],
    }


def _profile_event_recall_audit(
    *,
    truth_active: np.ndarray,
    alarm_active: np.ndarray,
    time_s: np.ndarray,
    step_dt_s: np.ndarray,
    effect_time_s: np.ndarray,
    episode_ids: list[str | int | float | bool],
    profile_ids: list[str | int | float | bool],
    detection_eligible: np.ndarray | None,
    minimum_event_recall: float | None,
) -> dict[str, Any]:
    """Audit event recall independently for every validation profile.

    Profiles without an eligible target event have no recall denominator. They
    are therefore excluded from the profile-recall constraint rather than
    being treated as either perfect or failed, and are retained explicitly in
    the returned audit.
    """

    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    profiles_with_events: list[Any] = []
    profiles_without_events: list[Any] = []
    rejected_profiles: list[Any] = []
    profile_order = _stable_json_unique(profile_ids)
    profile_array = np.asarray(
        [json.dumps(value, ensure_ascii=False) for value in profile_ids], dtype=object
    )
    episode_array = np.asarray(episode_ids, dtype=object)

    for profile_id in profile_order:
        key = json.dumps(profile_id, ensure_ascii=False)
        mask = profile_array == key
        metrics = event_detection_metrics(
            truth_active[mask],
            alarm_active[mask],
            time_s[mask],
            step_dt_s[mask],
            effect_time_s=effect_time_s[mask],
            episode_ids=episode_array[mask],
            detection_eligible=(
                None if detection_eligible is None else detection_eligible[mask]
            ),
        )
        target_events = int(metrics["target_events"])
        has_eligible_events = target_events > 0
        recall = metrics["event_recall"]
        if has_eligible_events:
            # event_detection_metrics guarantees a defined recall whenever the
            # target-event denominator is non-zero.
            if recall is None:  # pragma: no cover - defensive contract guard
                raise RuntimeError(
                    "event recall is undefined despite eligible target events"
                )
            scalar_recall = float(recall)
            recalls.append(scalar_recall)
            profiles_with_events.append(profile_id)
            constraint_satisfied = (
                None
                if minimum_event_recall is None
                else scalar_recall >= minimum_event_recall
            )
            rejection_reasons = (
                []
                if constraint_satisfied is not False
                else ["minimum_profile_event_recall_not_met"]
            )
            if rejection_reasons:
                rejected_profiles.append(profile_id)
            exclusion_reason = None
        else:
            profiles_without_events.append(profile_id)
            constraint_satisfied = None
            rejection_reasons = []
            exclusion_reason = "no_eligible_truth_events"

        rows.append(
            {
                "profile_id": profile_id,
                "eligible_target_events": target_events,
                "true_positive_events": int(metrics["true_positive_events"]),
                "false_negative_events": int(metrics["false_negative_events"]),
                "event_recall": recall,
                "constraint_applicable": has_eligible_events,
                "constraint_satisfied": constraint_satisfied,
                "constraint_exclusion_reason": exclusion_reason,
                "rejection_reasons": rejection_reasons,
            }
        )

    constraint_satisfied = (
        True
        if minimum_event_recall is None
        else not rejected_profiles
    )
    return {
        "aggregation": "minimum recall across profiles with eligible target events",
        "minimum_event_recall_limit": minimum_event_recall,
        "minimum_event_recall_operator": (
            None if minimum_event_recall is None else ">="
        ),
        "profiles": rows,
        "profiles_with_eligible_events": profiles_with_events,
        "profiles_without_eligible_events": profiles_without_events,
        "profiles_without_eligible_events_are_excluded": True,
        "eligible_profile_count": len(profiles_with_events),
        "excluded_profile_count": len(profiles_without_events),
        "minimum_profile_event_recall": min(recalls) if recalls else None,
        "constraint_satisfied": constraint_satisfied,
        "rejected_profiles": rejected_profiles,
    }


def _candidate_rank(candidate: Mapping[str, Any], objective: str) -> tuple[float, ...]:
    """Return a min-sort key implementing the documented deterministic ties."""

    event = candidate["event_metrics"]
    profile = candidate["profile_false_alarm_audit"]
    config = candidate["config"]
    objective_value = float(candidate["objective_value"])
    worst_rate = float(profile["maximum_profile_false_alarm_onsets_per_negative_hour"])
    pooled_rate = float(profile["pooled_false_alarm_onsets_per_negative_hour"])
    precision = event["event_precision"]
    precision_value = -1.0 if precision is None else float(precision)
    # Under equal validation behavior, prefer settings that are less prone to
    # asserting and quicker to clear, then use the remaining parameters.
    return (
        -objective_value,
        worst_rate,
        pooled_rate,
        -precision_value,
        float(candidate["alarm_active_duration_s"]),
        -float(config["on_threshold"]),
        -float(config["off_threshold"]),
        -float(config["min_on_duration_s"]),
        float(config["min_off_duration_s"]),
        -float(config["cooldown_s"]),
        0.0 if objective == "event_f1" else 1.0,
    )


def fit_alarm_policy(
    *,
    scores: Sequence[float] | np.ndarray,
    truth_active: Sequence[object] | np.ndarray,
    time_s: Sequence[float] | np.ndarray,
    step_dt_s: Sequence[float] | np.ndarray,
    effect_time_s: Sequence[float] | np.ndarray,
    episode_ids: Sequence[object] | np.ndarray,
    profile_ids: Sequence[object] | np.ndarray,
    splits: Sequence[object] | np.ndarray,
    on_thresholds: Sequence[float],
    off_thresholds: Sequence[float],
    min_on_durations_s: Sequence[float] = (0.0,),
    min_off_durations_s: Sequence[float] = (0.0,),
    cooldowns_s: Sequence[float] = (0.0,),
    max_false_alarm_onsets_per_negative_hour: float,
    min_event_recall: float | None = None,
    min_profile_event_recall: float | None = None,
    objective: str = "event_f1",
    detection_eligible: Sequence[object] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Select an alarm operating point using only ``validation_id`` rows.

    Candidates first have to satisfy the false-alarm limit independently for
    every profile. If ``min_profile_event_recall`` is configured, every profile
    with at least one eligible target event must also satisfy that recall
    floor; profiles without an eligible event are audited but excluded from
    its denominator. Among feasible candidates, ``objective`` is maximized.
    Undefined objectives (for example, no eligible validation events) are not
    selectable. The returned mapping is serializable with
    ``json.dumps(..., allow_nan=False)``.
    """

    split_array = _one_dimensional("splits", splits)
    split_values = [_json_scalar("splits", value) for value in split_array.tolist()]
    split_vocabulary = _stable_json_unique(split_values)
    if split_vocabulary != [_VALIDATION_SPLIT]:
        raise ValueError(
            "fit_alarm_policy is validation-only: split vocabulary must be "
            "exactly ['validation_id']"
        )

    score = _float_vector("scores", scores)
    n = int(score.size)
    if n == 0:
        raise ValueError("fit_alarm_policy requires at least one validation row")
    if int(split_array.size) != n:
        raise ValueError(f"splits must contain {n} values")
    truth = _binary_vector("truth_active", truth_active, length=n)
    times = _float_vector("time_s", time_s, length=n)
    durations = _float_vector(
        "step_dt_s", step_dt_s, length=n, strictly_positive=True
    )
    effects = np.asarray(effect_time_s, dtype=np.float64)
    if effects.ndim != 1 or int(effects.size) != n:
        raise ValueError(f"effect_time_s must contain {n} values")
    if np.isinf(effects).any():
        raise ValueError("effect_time_s may contain finite values or NaN, not infinity")
    episodes = _identifier_vector("episode_ids", episode_ids, length=n)
    profiles = _identifier_vector("profile_ids", profile_ids, length=n)
    eligible = (
        None
        if detection_eligible is None
        else _binary_vector("detection_eligible", detection_eligible, length=n)
    )
    max_rate = _finite_float(
        "max_false_alarm_onsets_per_negative_hour",
        max_false_alarm_onsets_per_negative_hour,
        non_negative=True,
    )
    if min_event_recall is None:
        recall_floor = None
    else:
        recall_floor = _finite_float(
            "min_event_recall", min_event_recall, non_negative=True
        )
        if recall_floor > 1.0:
            raise ValueError("min_event_recall must be between 0 and 1")
    if min_profile_event_recall is None:
        profile_recall_floor = None
    else:
        profile_recall_floor = _finite_float(
            "min_profile_event_recall",
            min_profile_event_recall,
            non_negative=True,
        )
        if profile_recall_floor > 1.0:
            raise ValueError("min_profile_event_recall must be between 0 and 1")
    if objective not in {"event_recall", "event_f1"}:
        raise ValueError("objective must be 'event_recall' or 'event_f1'")

    configs = _candidate_configs(
        on_thresholds=on_thresholds,
        off_thresholds=off_thresholds,
        min_on_durations_s=min_on_durations_s,
        min_off_durations_s=min_off_durations_s,
        cooldowns_s=cooldowns_s,
    )
    candidates: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for config in configs:
        alarm = apply_alarm_policy(score, durations, episodes, config)
        event = event_detection_metrics(
            truth,
            alarm,
            times,
            durations,
            effect_time_s=effects,
            episode_ids=episodes,
            detection_eligible=eligible,
        )
        profile_audit = _profile_false_alarm_audit(
            truth_active=truth,
            alarm_active=alarm,
            time_s=times,
            step_dt_s=durations,
            episode_ids=episodes,
            profile_ids=profiles,
        )
        profile_event_audit = _profile_event_recall_audit(
            truth_active=truth,
            alarm_active=alarm,
            time_s=times,
            step_dt_s=durations,
            effect_time_s=effects,
            episode_ids=episodes,
            profile_ids=profiles,
            detection_eligible=eligible,
            minimum_event_recall=profile_recall_floor,
        )
        worst_rate = profile_audit[
            "maximum_profile_false_alarm_onsets_per_negative_hour"
        ]
        objective_value = event[objective]
        has_exposure = bool(profile_audit["all_profiles_have_negative_exposure"])
        within_constraint = (
            has_exposure and worst_rate is not None and float(worst_rate) <= max_rate
        )
        event_recall = event["event_recall"]
        recall_floor_satisfied = (
            recall_floor is None
            or (
                event_recall is not None
                and float(event_recall) >= recall_floor
            )
        )
        profile_recall_floor_satisfied = bool(
            profile_event_audit["constraint_satisfied"]
        )
        selectable = (
            within_constraint
            and recall_floor_satisfied
            and profile_recall_floor_satisfied
            and objective_value is not None
        )
        candidate = {
            "config": config.to_dict(),
            "feasible": selectable,
            "constraint_satisfied": within_constraint,
            "objective": objective,
            "objective_value": objective_value,
            "event_metrics": event,
            "profile_false_alarm_audit": profile_audit,
            "profile_event_recall_audit": profile_event_audit,
            "alarm_active_duration_s": float(durations[alarm].sum()),
            "alarm_active_fraction": float(durations[alarm].sum() / durations.sum()),
            "rejection_reasons": [
                reason
                for condition, reason in (
                    (has_exposure, "profile_without_negative_exposure"),
                    (
                        has_exposure
                        and worst_rate is not None
                        and float(worst_rate) <= max_rate,
                        "profile_false_alarm_constraint_exceeded",
                    ),
                    (
                        recall_floor_satisfied,
                        "minimum_event_recall_not_met",
                    ),
                    (
                        profile_recall_floor_satisfied,
                        "minimum_profile_event_recall_not_met",
                    ),
                    (objective_value is not None, "undefined_validation_objective"),
                )
                if not condition
            ],
        }
        candidates.append(candidate)
        if selectable:
            feasible.append(candidate)

    selected = (
        min(feasible, key=lambda item: _candidate_rank(item, objective))
        if feasible
        else None
    )
    if selected is None:
        reasons = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate["rejection_reasons"]
            }
        )
        status = "no_operating_point"
    else:
        reasons = []
        status = "selected"

    result = {
        "version": ALARM_POLICY_VERSION,
        "status": status,
        "config": None if selected is None else selected["config"],
        "audit": {
            "fit_split": _VALIDATION_SPLIT,
            "split_vocabulary": split_vocabulary,
            "validation_rows_only": True,
            "rows": n,
            "profile_count": len(_stable_json_unique(profiles)),
            "episode_count": len(_stable_json_unique(episodes)),
            "objective": objective,
            "constraint": {
                "metric": "maximum_profile_false_alarm_onsets_per_negative_hour",
                "operator": "<=",
                "limit": max_rate,
                "requires_negative_exposure_for_every_profile": True,
                "minimum_event_recall": recall_floor,
                "minimum_event_recall_operator": (
                    None if recall_floor is None else ">="
                ),
                "minimum_profile_event_recall": profile_recall_floor,
                "minimum_profile_event_recall_operator": (
                    None if profile_recall_floor is None else ">="
                ),
                "profile_event_recall_scope": (
                    "each validation profile with at least one eligible target event"
                ),
                "profiles_without_eligible_events_are_excluded": True,
            },
            "candidate_count": len(candidates),
            "feasible_candidate_count": len(feasible),
            "no_operating_point_reasons": reasons,
            "tie_break_order": [
                f"maximize {objective}",
                "minimize maximum per-profile false-alarm-onset rate",
                "minimize pooled false-alarm-onset rate",
                "maximize event precision",
                "minimize alarm-active duration",
                "prefer higher on threshold",
                "prefer higher off threshold",
                "prefer longer assertion persistence",
                "prefer shorter clear persistence",
                "prefer longer cooldown",
            ],
            "selected_validation_metrics": (
                None
                if selected is None
                else {
                    "event_metrics": selected["event_metrics"],
                    "profile_false_alarm_audit": selected[
                        "profile_false_alarm_audit"
                    ],
                    "profile_event_recall_audit": selected[
                        "profile_event_recall_audit"
                    ],
                    "alarm_active_duration_s": selected[
                        "alarm_active_duration_s"
                    ],
                    "alarm_active_fraction": selected["alarm_active_fraction"],
                }
            ),
            "candidates": candidates,
            "claim_boundary": (
                "research operating-point selection only; no product-safety, "
                "HACCP-compliance, or release claim"
            ),
        },
    }
    # Assert the public contract here, not only in tests.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "ALARM_POLICY_VERSION",
    "AlarmPolicyConfig",
    "apply_alarm_policy",
    "fit_alarm_policy",
]
