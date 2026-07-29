"""Dependency-light, auditable metrics for the FlowTwin benchmark.

All public functions return JSON-safe dictionaries and use ``None`` when a
rate has no mathematical denominator.  In particular, an empty evaluation
population is never reported as perfect performance.

Time-series rows represent half-open intervals ``[time_s, time_s + dt_s)``.
Rows may be interleaved across episodes; they are grouped and ordered before
event construction.  Overlapping rows within an episode are rejected.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np


METRICS_VERSION = "0.2.0"
_TIME_ATOL = 1e-9


def _float_vector(
    name: str,
    values: Sequence[float] | np.ndarray,
    *,
    length: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must contain {length} values")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_vector(
    name: str,
    values: Sequence[object] | np.ndarray,
    *,
    length: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must contain {length} values")
    if array.dtype.kind in "fc":
        numeric = np.asarray(array, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain only binary 0/1 values")
    elif array.dtype.kind in "iu":
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must contain only binary 0/1 values")
    elif array.dtype.kind != "b":
        # Object/string coercion is intentionally rejected: bool("0") is True.
        raise ValueError(f"{name} must contain booleans or numeric binary values")
    return array.astype(np.bool_, copy=False)


def _json_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _validated_labels(name: str, values: Sequence[object] | np.ndarray) -> list[object]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    result: list[object] = []
    for raw in array.tolist():
        value = _json_scalar(raw)
        if value is None:
            raise ValueError(f"{name} may not contain None")
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError(f"{name} may not contain non-finite labels")
        if not isinstance(value, Hashable):
            raise ValueError(f"{name} labels must be hashable")
        result.append(value)
    return result


def _stable_unique(values: Iterable[object]) -> list[object]:
    result: list[object] = []
    seen: set[object] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else None


def classification_metrics(
    y_true: Sequence[object] | np.ndarray,
    y_pred: Sequence[object] | np.ndarray,
    *,
    labels: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Return multiclass confusion counts and macro-F1.

    ``macro_f1`` averages every class with non-zero true or predicted support.
    A label absent from both truth and prediction is retained in a requested
    confusion matrix but has ``f1=None`` and does not alter the average.
    """

    truth = _validated_labels("y_true", y_true)
    prediction = _validated_labels("y_pred", y_pred)
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have equal lengths")
    class_labels = (
        _validated_labels("labels", labels)
        if labels is not None
        else _stable_unique([*truth, *prediction])
    )
    if len(_stable_unique(class_labels)) != len(class_labels):
        raise ValueError("labels must not contain duplicates")
    class_to_index = {value: index for index, value in enumerate(class_labels)}
    unknown = _stable_unique(
        value for value in [*truth, *prediction] if value not in class_to_index
    )
    if unknown:
        raise ValueError(f"observations contain labels outside labels: {unknown!r}")

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction, strict=True):
        confusion[class_to_index[actual], class_to_index[predicted]] += 1

    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for index, label in enumerate(class_labels):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        fp = predicted - tp
        fn = support - tp
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _f1(tp, fp, fn)
        if f1 is not None:
            f1_values.append(f1)
        per_class.append(
            {
                "label": _json_scalar(label),
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    rows = len(truth)
    return {
        "version": METRICS_VERSION,
        "rows": rows,
        "accuracy": _safe_ratio(int(np.trace(confusion)), rows),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "labels": [_json_scalar(value) for value in class_labels],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "macro_definition": "unweighted mean over classes with true or predicted support",
    }


def _episode_vector(values: Sequence[object] | np.ndarray | None, length: int) -> list[object]:
    if values is None:
        return ["__all__"] * length
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.size != length:
        raise ValueError(f"episode_ids must contain {length} values")
    result: list[object] = []
    for value in array.tolist():
        value = _json_scalar(value)
        if value is None or not isinstance(value, Hashable):
            raise ValueError("episode_ids must contain non-null hashable values")
        result.append(value)
    return result


def _ordered_episode_indices(
    time_s: np.ndarray, dt_s: np.ndarray, episode_ids: list[object]
) -> list[tuple[object, list[int]]]:
    grouped: dict[object, list[int]] = {}
    for index, episode_id in enumerate(episode_ids):
        grouped.setdefault(episode_id, []).append(index)
    result: list[tuple[object, list[int]]] = []
    for episode_id, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: (time_s[index], index))
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = time_s[previous] + dt_s[previous]
            if time_s[current] < previous_end - _TIME_ATOL:
                raise ValueError(
                    f"time intervals overlap within episode {episode_id!r}"
                )
        result.append((episode_id, ordered))
    return result


def _time_inputs(
    time_s: Sequence[float] | np.ndarray,
    step_dt_s: Sequence[float] | np.ndarray,
    episode_ids: Sequence[object] | np.ndarray | None,
    length: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[object, list[int]]]]:
    times = _float_vector("time_s", time_s, length=length)
    durations = _float_vector("step_dt_s", step_dt_s, length=length)
    if np.any(durations <= 0.0):
        raise ValueError("step_dt_s must contain only positive durations")
    episodes = _episode_vector(episode_ids, length)
    return times, durations, _ordered_episode_indices(times, durations, episodes)


def _segments(
    mask: np.ndarray,
    ordered: list[int],
    times: np.ndarray,
    durations: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    previous_index: int | None = None
    for index in ordered:
        contiguous = (
            previous_index is not None
            and abs(times[index] - (times[previous_index] + durations[previous_index]))
            <= _TIME_ATOL
        )
        if bool(mask[index]):
            if active is None or not contiguous:
                if active is not None:
                    result.append(active)
                active = {
                    "start_s": float(times[index]),
                    "end_s": float(times[index] + durations[index]),
                    "indices": [index],
                }
            else:
                active["end_s"] = float(times[index] + durations[index])
                active["indices"].append(index)
        elif active is not None:
            result.append(active)
            active = None
        previous_index = index
    if active is not None:
        result.append(active)
    return result


def event_detection_metrics(
    truth_active: Sequence[object] | np.ndarray,
    alarm_active: Sequence[object] | np.ndarray,
    time_s: Sequence[float] | np.ndarray,
    step_dt_s: Sequence[float] | np.ndarray,
    *,
    effect_time_s: Sequence[float] | np.ndarray,
    episode_ids: Sequence[object] | np.ndarray | None = None,
    detection_eligible: Sequence[object] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one-to-one event detection from the observable-effect time.

    A truth event is a contiguous ``truth_active`` interval whose rows are
    detection eligible.  Its eligible rows must carry one finite, constant
    ``effect_time_s`` inside that event.  A predicted event is a contiguous
    alarm interval.  Chronological one-to-one matching requires overlap with
    ``[effect_time_s, truth_event_end)`` in the same episode, so one latched
    alarm cannot claim multiple target events.  An alarm already active at the
    observable-effect time receives zero latency.
    """

    truth = _binary_vector("truth_active", truth_active)
    n = int(truth.size)
    alarm = _binary_vector("alarm_active", alarm_active, length=n)
    eligible = (
        np.ones(n, dtype=np.bool_)
        if detection_eligible is None
        else _binary_vector("detection_eligible", detection_eligible, length=n)
    )
    effects = _float_vector(
        "effect_time_s", effect_time_s, length=n, finite=False
    )
    times, durations, episode_groups = _time_inputs(
        time_s, step_dt_s, episode_ids, n
    )

    target_events: list[dict[str, Any]] = []
    predicted_events: list[dict[str, Any]] = []
    for episode_id, ordered in episode_groups:
        raw_truth_segments = _segments(truth, ordered, times, durations)
        for segment in raw_truth_segments:
            flags = eligible[np.asarray(segment["indices"], dtype=np.int64)]
            if np.any(flags) and not np.all(flags):
                raise ValueError(
                    "detection_eligible must be constant within each truth event"
                )
            if not bool(flags[0]):
                continue
            event_effects = effects[np.asarray(segment["indices"], dtype=np.int64)]
            if not np.isfinite(event_effects).all():
                raise ValueError(
                    "eligible truth events require finite effect_time_s on every row"
                )
            reference = float(event_effects[0])
            if not np.allclose(event_effects, reference, atol=_TIME_ATOL, rtol=0.0):
                raise ValueError(
                    "effect_time_s must be constant within each eligible truth event"
                )
            if reference < float(segment["start_s"]) - _TIME_ATOL:
                raise ValueError("effect_time_s may not precede its truth event")
            if reference >= float(segment["end_s"]) - _TIME_ATOL:
                raise ValueError("effect_time_s must leave observable event duration")
            target_events.append(
                {**segment, "episode_id": episode_id, "effect_s": reference}
            )
        for segment in _segments(alarm, ordered, times, durations):
            predicted_events.append({**segment, "episode_id": episode_id})

    target_events.sort(key=lambda item: (str(item["episode_id"]), item["effect_s"]))
    predicted_events.sort(key=lambda item: (str(item["episode_id"]), item["start_s"]))
    unmatched_predictions = set(range(len(predicted_events)))
    latencies: list[float] = []
    horizon_penalized_latencies: list[float] = []
    matched = 0
    for target in target_events:
        candidates = [
            index
            for index in unmatched_predictions
            if predicted_events[index]["episode_id"] == target["episode_id"]
            and float(predicted_events[index]["end_s"])
            > float(target["effect_s"]) + _TIME_ATOL
            and float(predicted_events[index]["start_s"])
            < float(target["end_s"]) - _TIME_ATOL
        ]
        if not candidates:
            horizon_penalized_latencies.append(
                max(0.0, float(target["end_s"]) - float(target["effect_s"]))
            )
            continue
        selected = min(candidates, key=lambda index: predicted_events[index]["start_s"])
        prediction = predicted_events[selected]
        detection_time = max(float(target["effect_s"]), float(prediction["start_s"]))
        latency = max(0.0, detection_time - float(target["effect_s"]))
        latencies.append(latency)
        horizon_penalized_latencies.append(latency)
        unmatched_predictions.remove(selected)
        matched += 1

    target_count = len(target_events)
    prediction_count = len(predicted_events)
    fp = prediction_count - matched
    fn = target_count - matched
    latency_array = np.asarray(latencies, dtype=np.float64)
    penalized_array = np.asarray(horizon_penalized_latencies, dtype=np.float64)
    return {
        "version": METRICS_VERSION,
        "target_events": target_count,
        "predicted_events": prediction_count,
        "true_positive_events": matched,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "event_precision": _safe_ratio(matched, prediction_count),
        "event_recall": _safe_ratio(matched, target_count),
        "event_f1": _f1(matched, fp, fn),
        "detection_latencies_s": [float(value) for value in latencies],
        "detection_latency_mean_s": (
            float(latency_array.mean()) if latency_array.size else None
        ),
        "detection_latency_median_s": (
            float(np.median(latency_array)) if latency_array.size else None
        ),
        "detection_latency_p90_s": (
            float(np.quantile(latency_array, 0.9)) if latency_array.size else None
        ),
        "detection_latency_max_s": (
            float(latency_array.max()) if latency_array.size else None
        ),
        "horizon_penalized_latency_mean_s": (
            float(penalized_array.mean()) if penalized_array.size else None
        ),
        "horizon_penalized_latency_median_s": (
            float(np.median(penalized_array)) if penalized_array.size else None
        ),
        "horizon_penalized_latency_values_s": [
            float(value) for value in horizon_penalized_latencies
        ],
        "miss_penalty_policy": (
            "observable target-event horizon from effect_time_s to event end"
        ),
        "latency_reference": "observable_effect_time_s",
        "matching": "chronological one-to-one interval overlap within episode",
    }


def false_alarm_metrics(
    truth_active: Sequence[object] | np.ndarray,
    alarm_active: Sequence[object] | np.ndarray,
    time_s: Sequence[float] | np.ndarray,
    step_dt_s: Sequence[float] | np.ndarray,
    *,
    episode_ids: Sequence[object] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return nuisance-alarm onsets and duration per negative operating hour.

    A false-alarm onset is the start of a maximal contiguous
    ``alarm_active and not truth_active`` interval.  Consequently, a latched
    alarm that persists beyond a true event starts a new false interval at the
    true-to-negative boundary.  Gaps and episode boundaries split intervals.
    """

    truth = _binary_vector("truth_active", truth_active)
    n = int(truth.size)
    alarm = _binary_vector("alarm_active", alarm_active, length=n)
    times, durations, episode_groups = _time_inputs(
        time_s, step_dt_s, episode_ids, n
    )
    negative = ~truth
    false_mask = alarm & negative
    negative_duration_s = float(durations[negative].sum())
    false_duration_s = float(durations[false_mask].sum())
    onset_count = 0
    for _episode_id, ordered in episode_groups:
        onset_count += len(_segments(false_mask, ordered, times, durations))
    negative_hours = negative_duration_s / 3600.0
    return {
        "version": METRICS_VERSION,
        "rows": n,
        "negative_duration_s": negative_duration_s,
        "false_alarm_onsets": onset_count,
        "false_alarm_duration_s": false_duration_s,
        "false_alarm_onsets_per_negative_hour": _safe_ratio(
            onset_count, negative_hours
        ),
        "false_alarm_duration_s_per_negative_hour": _safe_ratio(
            false_duration_s, negative_hours
        ),
        "false_alarm_duration_fraction": _safe_ratio(
            false_duration_s, negative_duration_s
        ),
        "denominator": "truth-negative operating duration",
    }


def volume_metrics(
    unsafe_forward_l: Sequence[float] | np.ndarray | None,
    false_divert_l: Sequence[float] | np.ndarray | None,
    cooling_oos_l: Sequence[float] | np.ndarray | None,
) -> dict[str, Any]:
    """Sum mutually explicit, incremental process-volume consequences.

    Inputs are per-row litres, not cumulative meters or Boolean labels.  Pass
    ``None`` when a consequence is not available for an experiment; its total
    is then explicitly reported as ``None`` with status ``not_applicable``.
    Available vectors must all have the same row count.  The
    caller must construct ``false_divert_l`` against its declared routing
    oracle and ``cooling_oos_l`` against its declared outlet-temperature limit;
    this utility deliberately does not guess either policy.
    """

    raw = {
        "unsafe_forward": unsafe_forward_l,
        "false_divert": false_divert_l,
        "cooling_oos": cooling_oos_l,
    }
    arrays: dict[str, np.ndarray] = {}
    n: int | None = None
    for name, values in raw.items():
        if values is None:
            continue
        array = _float_vector(f"{name}_l", values, length=n)
        if n is None:
            n = int(array.size)
        if np.any(array < 0.0):
            raise ValueError(
                f"{name}_l must contain non-negative incremental volumes"
            )
        arrays[name] = array

    def total(name: str) -> float | None:
        return float(arrays[name].sum()) if name in arrays else None

    return {
        "version": METRICS_VERSION,
        "rows": n,
        "unsafe_forward_volume_l": total("unsafe_forward"),
        "false_divert_volume_l": total("false_divert"),
        "cooling_oos_volume_l": total("cooling_oos"),
        "status": {
            name: "reported" if name in arrays else "not_applicable"
            for name in raw
        },
        "input_semantics": "non-negative incremental litres per evaluated row",
    }


def _prediction_sets(
    values: Sequence[Iterable[object]] | np.ndarray, length: int
) -> list[set[object]]:
    if isinstance(values, np.ndarray) and values.ndim == 2:
        if values.shape[0] != length:
            raise ValueError(f"prediction_sets must contain {length} rows")
        membership = _binary_vector(
            "prediction_sets", values.reshape(-1), length=int(values.size)
        ).reshape(values.shape)
        return [set(np.flatnonzero(row).tolist()) for row in membership]
    try:
        rows = list(values)
    except TypeError as error:
        raise ValueError("prediction_sets must be a sequence of class iterables") from error
    if len(rows) != length:
        raise ValueError(f"prediction_sets must contain {length} rows")
    result: list[set[object]] = []
    for row_index, raw_row in enumerate(rows):
        if isinstance(raw_row, (str, bytes)):
            raise ValueError("each prediction set must be a non-string iterable")
        try:
            labels = _validated_labels(f"prediction_sets[{row_index}]", list(raw_row))
        except TypeError as error:
            raise ValueError("each prediction set must be iterable") from error
        if len(_stable_unique(labels)) != len(labels):
            raise ValueError("prediction sets may not contain duplicate labels")
        result.append(set(labels))
    return result


def conformal_metrics(
    y_true: Sequence[object] | np.ndarray,
    prediction_sets: Sequence[Iterable[object]] | np.ndarray,
) -> dict[str, Any]:
    """Return marginal prediction-set coverage and size statistics."""

    truth = _validated_labels("y_true", y_true)
    sets = _prediction_sets(prediction_sets, len(truth))
    sizes = np.asarray([len(value) for value in sets], dtype=np.int64)
    covered = np.asarray(
        [target in prediction_set for target, prediction_set in zip(truth, sets, strict=True)],
        dtype=np.bool_,
    )
    n = len(truth)
    return {
        "version": METRICS_VERSION,
        "rows": n,
        "empirical_coverage": float(covered.mean()) if n else None,
        "miscoverage_count": int(n - int(covered.sum())),
        "mean_prediction_set_size": float(sizes.mean()) if n else None,
        "median_prediction_set_size": float(np.median(sizes)) if n else None,
        "min_prediction_set_size": int(sizes.min()) if n else None,
        "max_prediction_set_size": int(sizes.max()) if n else None,
        "empty_set_fraction": float(np.mean(sizes == 0)) if n else None,
        "singleton_fraction": float(np.mean(sizes == 1)) if n else None,
        "coverage_definition": "fraction of rows whose prediction set contains y_true",
    }


def _risk_arrays(
    losses: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_losses = losses[order]
    cumulative_loss = np.cumsum(sorted_losses, dtype=np.float64)
    # Only thresholds after the last member of a tie are operationally attainable.
    tie_ends = np.flatnonzero(
        np.r_[sorted_confidence[:-1] != sorted_confidence[1:], True]
    )
    n = losses.size
    accepted = tie_ends + 1
    return (
        accepted.astype(np.float64) / n,
        cumulative_loss[tie_ends] / accepted,
        sorted_confidence[tie_ends],
        accepted,
    )


def _curve_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    selected = np.rint(np.linspace(0, length - 1, max_points)).astype(np.int64)
    return np.unique(selected)


def risk_coverage_curve(
    losses: Sequence[float] | np.ndarray,
    confidence_scores: Sequence[float] | np.ndarray,
    *,
    max_points: int = 101,
) -> dict[str, Any]:
    """Return selective risk as low-confidence rows are abstained.

    Higher ``confidence_scores`` are accepted first.  Points are emitted only
    at complete score-tie boundaries and are deterministically downsampled to
    at most ``max_points``.  AURC is computed from every attainable operating
    point before downsampling, with the first attainable risk extended to zero
    coverage.  Losses may be binary errors or any finite non-negative loss.
    """

    loss = _float_vector("losses", losses)
    confidence = _float_vector(
        "confidence_scores", confidence_scores, length=int(loss.size)
    )
    if np.any(loss < 0.0):
        raise ValueError("losses must be non-negative")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer")
    max_points = int(max_points)
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if not loss.size:
        return {
            "version": METRICS_VERSION,
            "rows": 0,
            "area_under_risk_coverage": None,
            "full_coverage_risk": None,
            "minimum_selective_risk": None,
            "points": [],
            "confidence_direction": "higher score is more confident",
        }
    coverages, risks, thresholds, accepted = _risk_arrays(loss, confidence)
    area = float(
        np.trapezoid(
            np.r_[risks[0], risks],
            np.r_[0.0, coverages],
        )
    )
    selected = _curve_indices(int(coverages.size), max_points)
    points = [
        {
            "coverage": float(coverages[index]),
            "risk": float(risks[index]),
            "threshold": float(thresholds[index]),
            "accepted": int(accepted[index]),
        }
        for index in selected
    ]
    return {
        "version": METRICS_VERSION,
        "rows": int(loss.size),
        "area_under_risk_coverage": area,
        "full_coverage_risk": float(loss.mean()),
        "minimum_selective_risk": float(risks.min()),
        "points": points,
        "confidence_direction": "higher score is more confident",
        "tie_policy": "operate only after complete confidence-score tie groups",
    }


def ood_detection_metrics(
    is_ood: Sequence[object] | np.ndarray,
    ood_scores: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Return tie-aware AUROC, average precision (AUPR), and FPR at 95% TPR.

    Larger scores must indicate stronger OOD evidence.  Metrics requiring both
    classes are ``None`` when the evaluation population lacks either class.
    AUPR is grouped-threshold average precision, not trapezoidal PR area.
    """

    truth = _binary_vector("is_ood", is_ood)
    scores = _float_vector("ood_scores", ood_scores, length=int(truth.size))
    n = int(truth.size)
    positives = int(truth.sum())
    negatives = n - positives
    auroc: float | None = None
    aupr: float | None = None
    fpr95: float | None = None
    if positives and negatives:
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_truth = truth[order].astype(np.int64)
        tp_cumulative = np.cumsum(sorted_truth)
        fp_cumulative = np.cumsum(1 - sorted_truth)
        tie_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
        tp = tp_cumulative[tie_ends].astype(np.float64)
        fp = fp_cumulative[tie_ends].astype(np.float64)
        tpr = tp / positives
        fpr = fp / negatives
        auroc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
        precision = tp / (tp + fp)
        recall_previous = np.r_[0.0, tpr[:-1]]
        aupr = float(np.sum((tpr - recall_previous) * precision))
        reached = np.flatnonzero(tpr >= 0.95)
        fpr95 = float(fpr[reached].min()) if reached.size else None
    return {
        "version": METRICS_VERSION,
        "rows": n,
        "positives": positives,
        "negatives": negatives,
        "prevalence": _safe_ratio(positives, n),
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "score_direction": "higher score indicates OOD",
        "aupr_definition": "grouped-threshold average precision",
    }


__all__ = [
    "METRICS_VERSION",
    "classification_metrics",
    "conformal_metrics",
    "event_detection_metrics",
    "false_alarm_metrics",
    "ood_detection_metrics",
    "risk_coverage_curve",
    "volume_metrics",
]
