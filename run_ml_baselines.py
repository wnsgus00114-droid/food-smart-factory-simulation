#!/usr/bin/env python3
"""Run rule, EWMA and CUSUM baselines with event-level HTST evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ml_pipeline_common import (
    BASE_DIR,
    CONTRACT_PATH,
    load_contract,
    percentile,
    read_csv_header,
    read_csv_rows,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_csv_header,
    write_json,
)


BASELINE_VERSION = "2.2.0"
METHODS = ("rule", "ewma", "cusum")
NORMAL_BEHAVIOR_CODES = frozenset({"N00", "M01", "M02"})
EXPLICIT_FORBIDDEN = {
    "alarm_count",
    "observable_alarm_count",
    "alarm_unsafe_forward",
    "diagnostic_alarm_count",
    "oracle_alarm_count",
    "total_alarm_count",
    "differential_pressure_bar",
    "fdv_actual_forward",
    "power_available",
    "sensor_available",
    "valve_travel_time_factor",
    "valve_leakage_fraction",
    "cleaning_effectiveness_factor",
}
MONITORED_FEATURES = (
    "control_temp_sensor_c",
    "safety_temp_sensor_c",
    "measured_flow_l_h",
    "steam_valve",
    "product_temp_sensor_c",
    "raw_pressure_sensor_bar",
    "pasteurized_pressure_sensor_bar",
    "measured_differential_pressure_bar",
    "booster_pump_speed_fraction",
    "leak_detector_signal_fraction",
    "preheat_temp_sensor_c",
    "sensor_disagreement_c",
    "estimated_residence_time_s",
    "estimated_fastest_residence_time_s",
    "fdv_command_position",
    "fdv_position_feedback",
    "fdv_position_error",
    "cip_chemical_concentration_pct",
    "cip_conductivity_proxy_ms_cm",
    "cip_ph_proxy",
    "power_good_signal",
    "temperature_sensor_quality_ok",
)
RULE_INPUTS = (
    "safety_temp_sensor_c",
    "estimated_fastest_residence_time_s",
    "measured_flow_l_h",
    "maximum_safe_flow_l_h",
    "measured_differential_pressure_bar",
    "sensor_disagreement_c",
    "leak_detector_signal_fraction",
    "fdv_position_feedback",
    "product_temp_sensor_c",
    "cip_chemical_concentration_pct",
    "cip_conductivity_proxy_ms_cm",
    "cip_ph_proxy",
    "power_good_signal",
    "temperature_sensor_quality_ok",
)
OPERATIONAL_INPUTS = ("cip_cycle_active", "step_dt_s")
BASELINE_INPUTS = tuple(
    dict.fromkeys((*MONITORED_FEATURES, *RULE_INPUTS, *OPERATIONAL_INPUTS))
)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(0.0, self.m2 / (self.count - 1)))


class Reservoir:
    """Deterministic bounded reservoir for validation score quantiles."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self.seen = 0
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.size:
            self.values.append(value)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.values[index] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ewma-alpha", type=float, default=0.20)
    parser.add_argument("--cusum-drift", type=float, default=0.50)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--reservoir-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def _prepare_output(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def _joined_rows(dataset: Path) -> Iterator[tuple[dict[str, str], dict[str, str]]]:
    with (
        (dataset / "signals.csv").open(newline="", encoding="utf-8") as signal_handle,
        (dataset / "oracle_labels.csv").open(newline="", encoding="utf-8") as label_handle,
    ):
        signals = csv.DictReader(signal_handle)
        labels = csv.DictReader(label_handle)
        while True:
            signal = next(signals, None)
            label = next(labels, None)
            if signal is None or label is None:
                if signal is not None or label is not None:
                    raise ValueError("signals.csv and oracle_labels.csv have different lengths")
                return
            for key in ("episode_id", "time_start_s", "time_s"):
                if signal[key] != label[key]:
                    raise ValueError(f"Signal/label row misalignment at {key}")
            yield signal, label


def _fit_stats(
    dataset: Path,
    episode_meta: Mapping[str, Mapping[str, str]],
    episode_split: Mapping[str, str],
) -> dict[str, dict[str, RunningStats]]:
    stats: dict[str, dict[str, RunningStats]] = defaultdict(
        lambda: {feature: RunningStats() for feature in MONITORED_FEATURES}
    )
    accepted = 0
    for signal, _ in _joined_rows(dataset):
        episode = signal["episode_id"]
        if episode_split[episode] != "train_id":
            continue
        if episode_meta[episode]["canonical_code"] not in NORMAL_BEHAVIOR_CODES:
            continue
        regime = _operational_regime(signal)
        for feature in MONITORED_FEATURES:
            value = float(signal[feature])
            stats[regime][feature].add(value)
            stats["__GLOBAL__"][feature].add(value)
        accepted += 1
    if accepted == 0:
        raise ValueError(
            "No N00/M01/M02 normal-behavior rows exist in train_id for baseline fitting"
        )
    return dict(stats)


def _operational_regime(signal: Mapping[str, str]) -> str:
    """Derive the scoring regime solely from the observable CIP status bit."""
    return "cip" if float(signal["cip_cycle_active"]) >= 0.5 else "production"


def _standardized(
    signal: Mapping[str, str], stats: Mapping[str, Mapping[str, RunningStats]]
) -> dict[str, float]:
    regime_stats = stats.get(_operational_regime(signal), stats["__GLOBAL__"])
    global_stats = stats["__GLOBAL__"]
    z: dict[str, float] = {}
    for feature in MONITORED_FEATURES:
        item = regime_stats[feature]
        if item.count < 10:
            item = global_stats[feature]
        scale = max(item.std, 1e-6)
        z[feature] = (float(signal[feature]) - item.mean) / scale
    return z


def _rule_score(
    signal: Mapping[str, str],
    thresholds: Mapping[str, float],
    cip_chemical_elapsed_s: float,
) -> float:
    common_violations = (
        float(signal["power_good_signal"]) < 0.5,
        float(signal["temperature_sensor_quality_ok"]) < 0.5,
    )
    if _operational_regime(signal) == "cip":
        chemical_pct = float(signal["cip_chemical_concentration_pct"])
        cip_violations: tuple[bool, ...] = ()
        if (
            chemical_pct > 1e-12
            and cip_chemical_elapsed_s
            >= float(thresholds["cip_response_grace_s"])
        ):
            conductivity_floor = 0.2 + (
                0.45
                * chemical_pct
                * float(thresholds["minimum_cip_conductivity_fraction_of_command"])
            )
            ph = float(signal["cip_ph_proxy"])
            ph_responded = (
                ph >= float(thresholds["minimum_cip_caustic_ph"])
                or ph <= float(thresholds["maximum_cip_acid_ph"])
            )
            cip_violations = (
                float(signal["cip_conductivity_proxy_ms_cm"])
                < conductivity_floor,
                not ph_responded,
            )
        return float(any((*common_violations, *cip_violations)))

    production_violations = (
        float(signal["safety_temp_sensor_c"])
        < float(thresholds["minimum_safety_temperature_c"]),
        float(signal["estimated_fastest_residence_time_s"])
        < float(thresholds["minimum_residence_time_s"]),
        float(signal["measured_flow_l_h"])
        > float(signal["maximum_safe_flow_l_h"]),
        float(signal["measured_differential_pressure_bar"])
        < float(thresholds["minimum_differential_pressure_bar"]),
        float(signal["sensor_disagreement_c"])
        > float(thresholds["maximum_sensor_disagreement_c"]),
        float(signal["leak_detector_signal_fraction"])
        >= float(thresholds["leak_detector_alarm_fraction"]),
        float(signal["fdv_position_feedback"])
        > float(thresholds["fdv_forward_feedback_threshold"])
        and float(signal["product_temp_sensor_c"])
        > float(thresholds["maximum_product_temperature_c"]),
    )
    violations = (*common_violations, *production_violations)
    return float(any(violations))


def _scored_rows(
    dataset: Path,
    stats: Mapping[str, Mapping[str, RunningStats]],
    thresholds: Mapping[str, float],
    ewma_alpha: float,
    cusum_drift: float,
) -> Iterator[tuple[dict[str, str], dict[str, str], dict[str, float]]]:
    previous_episode: str | None = None
    previous_cip_active: bool | None = None
    previous_chemical_pct = 0.0
    cip_chemical_elapsed_s = 0.0
    ewma: dict[str, float] = {}
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}
    for signal, label in _joined_rows(dataset):
        episode = signal["episode_id"]
        cip_active = _operational_regime(signal) == "cip"
        chemical_pct = float(signal["cip_chemical_concentration_pct"])
        if episode != previous_episode or cip_active != previous_cip_active:
            ewma = {feature: 0.0 for feature in MONITORED_FEATURES}
            positive = {feature: 0.0 for feature in MONITORED_FEATURES}
            negative = {feature: 0.0 for feature in MONITORED_FEATURES}
            previous_episode = episode
            previous_cip_active = cip_active
            previous_chemical_pct = 0.0
            cip_chemical_elapsed_s = 0.0
        if cip_active and chemical_pct > 1e-12:
            if previous_chemical_pct <= 1e-12 or not math.isclose(
                chemical_pct, previous_chemical_pct, rel_tol=0.0, abs_tol=1e-12
            ):
                cip_chemical_elapsed_s = 0.0
            cip_chemical_elapsed_s += float(signal["step_dt_s"])
        else:
            cip_chemical_elapsed_s = 0.0
        previous_chemical_pct = chemical_pct
        z = _standardized(signal, stats)
        for feature, value in z.items():
            ewma[feature] = ewma_alpha * value + (1.0 - ewma_alpha) * ewma[feature]
            positive[feature] = max(0.0, positive[feature] + value - cusum_drift)
            negative[feature] = max(0.0, negative[feature] - value - cusum_drift)
        scores = {
            "rule": _rule_score(signal, thresholds, cip_chemical_elapsed_s),
            "ewma": max(abs(value) for value in ewma.values()),
            "cusum": max(
                max(positive[feature], negative[feature])
                for feature in MONITORED_FEATURES
            ),
        }
        yield signal, label, scores


def _calibrate(
    dataset: Path,
    stats: Mapping[str, Mapping[str, RunningStats]],
    rule_thresholds: Mapping[str, float],
    episode_split: Mapping[str, str],
    episode_meta: Mapping[str, Mapping[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    validation = {
        method: Reservoir(args.reservoir_size, args.seed + index)
        for index, method in enumerate(("ewma", "cusum"))
    }
    training = {
        method: Reservoir(args.reservoir_size, args.seed + 100 + index)
        for index, method in enumerate(("ewma", "cusum"))
    }
    accepted_by_code: dict[str, int] = defaultdict(int)
    for signal, label, scores in _scored_rows(
        dataset, stats, rule_thresholds, args.ewma_alpha, args.cusum_drift
    ):
        if label["fault_active"] != "0":
            continue
        episode_id = signal["episode_id"]
        canonical_code = episode_meta[episode_id]["canonical_code"]
        if canonical_code not in NORMAL_BEHAVIOR_CODES:
            continue
        split = episode_split[episode_id]
        target = (
            validation
            if split == "validation_id"
            else training
            if split == "train_id"
            else None
        )
        if target is None:
            continue
        accepted_by_code[canonical_code] += 1
        for method in target:
            target[method].add(scores[method])
    thresholds = {"rule": 0.5}
    calibration: dict[str, Any] = {}
    floors = {"ewma": 3.0, "cusum": 5.0}
    for method in ("ewma", "cusum"):
        candidates = (
            (validation[method], "validation_id_normal_behaviors"),
            (training[method], "train_id_normal_behaviors_fallback"),
        )
        reservoir, source = next(
            ((candidate, name) for candidate, name in candidates if candidate.values),
            (training[method], "unavailable"),
        )
        if not reservoir.values:
            raise ValueError(
                f"No N00/M01/M02 normal-behavior rows available to calibrate {method}"
            )
        raw = percentile(reservoir.values, args.quantile)
        thresholds[method] = max(floors[method], raw)
        calibration[method] = {
            "source": source,
            "quantile": args.quantile,
            "rows_seen": reservoir.seen,
            "reservoir_rows": len(reservoir.values),
            "raw_quantile": raw,
            "floor": floors[method],
            "threshold": thresholds[method],
        }
    calibration["rule"] = {
        "source": "fixed engineering thresholds from ml_contract.json",
        "threshold": thresholds["rule"],
    }
    calibration["population"] = {
        "normal_behavior_codes": sorted(NORMAL_BEHAVIOR_CODES),
        "accepted_rows_by_code": dict(sorted(accepted_by_code.items())),
        "operational_regime_source": "observable cip_cycle_active",
    }
    return thresholds, calibration


def _events(rows: Sequence[tuple[dict[str, str], dict[str, str], dict[str, float]]]) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    active: dict[str, float] | None = None
    for signal, label, _ in rows:
        if label.get("detection_eligible", "1") != "1":
            continue
        is_active = label["fault_active"] == "1"
        if is_active and active is None:
            effect_text = label["fault_effect_time_s"]
            active = {
                "start": float(signal["time_start_s"]),
                "end": float(signal["time_s"]),
                "effect": float(effect_text) if effect_text else float(signal["time_start_s"]),
            }
        elif is_active and active is not None:
            active["end"] = float(signal["time_s"])
        elif not is_active and active is not None:
            events.append(active)
            active = None
    if active is not None:
        events.append(active)
    return events


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _evaluate_episode(
    rows: Sequence[tuple[dict[str, str], dict[str, str], dict[str, float]]],
    episode_meta: Mapping[str, str],
    split: str,
    thresholds: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    events = _events(rows)
    output: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for method in METHODS:
        tp = fp = fn = tn = 0
        false_edges = 0
        false_production_edges = 0
        false_cip_edges = 0
        false_seconds = 0.0
        false_production_seconds = 0.0
        false_cip_seconds = 0.0
        negative_operational_seconds = 0.0
        negative_production_seconds = 0.0
        negative_cip_seconds = 0.0
        previous_alarm = False
        previous_alarm_by_regime = {"production": False, "cip": False}
        alarm_rows: list[tuple[float, float]] = []
        for signal, label, scores in rows:
            alarm = scores[method] >= thresholds[method]
            start = float(signal["time_start_s"])
            end = float(signal["time_s"])
            truth = label["fault_active"] == "1"
            if alarm:
                alarm_rows.append((start, end))
            # detection_eligible is intentionally confined to event/latency
            # construction in _events(). Point and false-alarm metrics cover
            # every row, including all normal N00/M01/M02 intervals.
            regime = _operational_regime(signal)
            if truth and alarm:
                tp += 1
            elif truth and not alarm:
                fn += 1
            elif not truth and alarm:
                fp += 1
            else:
                tn += 1
            if not truth:
                interval_s = end - start
                negative_operational_seconds += interval_s
                if regime == "production":
                    negative_production_seconds += interval_s
                else:
                    negative_cip_seconds += interval_s
                if alarm:
                    false_seconds += interval_s
                    if regime == "production":
                        false_production_seconds += interval_s
                    else:
                        false_cip_seconds += interval_s
                if alarm and not previous_alarm:
                    false_edges += 1
                if alarm and not previous_alarm_by_regime[regime]:
                    if regime == "production":
                        false_production_edges += 1
                    else:
                        false_cip_edges += 1
            previous_alarm = alarm
            previous_alarm_by_regime[regime] = alarm

        detected = 0
        latencies: list[float] = []
        unsafe_before_alarm = 0.0
        for event in events:
            detection_times = [
                start
                for start, end in alarm_rows
                if end > event["effect"] + 1e-12 and start < event["end"] - 1e-12
            ]
            first_alarm = min(detection_times) if detection_times else None
            if first_alarm is not None:
                detected += 1
                latencies.append(max(0.0, first_alarm - event["effect"]))
            cutoff = first_alarm if first_alarm is not None else event["end"]
            for signal, label, _ in rows:
                start = float(signal["time_start_s"])
                end = float(signal["time_s"])
                if (
                    end > event["effect"] + 1e-12
                    and start < min(cutoff, event["end"]) - 1e-12
                ):
                    unsafe_before_alarm += float(label["unsafe_forward_l"])

        metrics = {
            "episode_id": episode_meta["episode_id"],
            "split": split,
            "canonical_code": episode_meta["canonical_code"],
            "method": method,
            "target_events": len(events),
            "detected_events": detected,
            "first_detection_latency_s": min(latencies) if latencies else "",
            "false_alarm_edges": false_edges,
            "false_alarm_production_edges": false_production_edges,
            "false_alarm_cip_edges": false_cip_edges,
            "negative_operational_seconds": negative_operational_seconds,
            "negative_production_seconds": negative_production_seconds,
            "negative_cip_seconds": negative_cip_seconds,
            "false_alarm_seconds": false_seconds,
            "false_alarm_production_seconds": false_production_seconds,
            "false_alarm_cip_seconds": false_cip_seconds,
            "unsafe_forward_l_before_detection": unsafe_before_alarm,
            "point_tp": tp,
            "point_fp": fp,
            "point_fn": fn,
            "point_tn": tn,
        }
        output.append(metrics)
        aggregate[(method, split)] = {
            **metrics,
            "latencies": latencies,
        }
    return output, aggregate


def _merge_aggregate(
    target: dict[tuple[str, str], dict[str, Any]],
    source: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    summed = (
        "target_events",
        "detected_events",
        "false_alarm_edges",
        "false_alarm_production_edges",
        "false_alarm_cip_edges",
        "negative_operational_seconds",
        "negative_production_seconds",
        "negative_cip_seconds",
        "false_alarm_seconds",
        "false_alarm_production_seconds",
        "false_alarm_cip_seconds",
        "unsafe_forward_l_before_detection",
        "point_tp",
        "point_fp",
        "point_fn",
        "point_tn",
    )
    for key, values in source.items():
        item = target.setdefault(key, {field: 0.0 for field in summed} | {"latencies": []})
        for field in summed:
            item[field] += float(values[field])
        item["latencies"].extend(values["latencies"])


def _final_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
    tp, fp, fn = values["point_tp"], values["point_fp"], values["point_fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    latencies = list(values["latencies"])
    negative_hours = values["negative_operational_seconds"] / 3600.0
    negative_production_hours = values["negative_production_seconds"] / 3600.0
    negative_cip_hours = values["negative_cip_seconds"] / 3600.0
    return {
        "target_events": int(values["target_events"]),
        "detected_events": int(values["detected_events"]),
        "event_recall": _safe_div(values["detected_events"], values["target_events"]),
        "detection_latency_median_s": percentile(latencies, 0.5) if latencies else None,
        "detection_latency_p90_s": percentile(latencies, 0.9) if latencies else None,
        "false_alarm_edges_per_operating_hour": _safe_div(
            values["false_alarm_edges"], negative_hours
        ),
        "false_alarm_edges_per_production_hour": _safe_div(
            values["false_alarm_production_edges"], negative_production_hours
        ),
        "false_alarm_edges_per_cip_hour": _safe_div(
            values["false_alarm_cip_edges"], negative_cip_hours
        ),
        "false_alarm_fraction": _safe_div(
            values["false_alarm_seconds"], values["negative_operational_seconds"]
        ),
        "false_alarm_production_fraction": _safe_div(
            values["false_alarm_production_seconds"],
            values["negative_production_seconds"],
        ),
        "false_alarm_cip_fraction": _safe_div(
            values["false_alarm_cip_seconds"], values["negative_cip_seconds"]
        ),
        "unsafe_forward_l_before_detection": values[
            "unsafe_forward_l_before_detection"
        ],
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": f1,
        "point_confusion": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(values["point_tn"]),
        },
        "evaluation_population": "all rows for point/false-alarm metrics, partitioned by observable cip_cycle_active; detection-eligible contiguous fault_active intervals for event/latency metrics (ineligible positive rows remain in point metrics)",
    }


def run(args: argparse.Namespace) -> Path:
    if not 0.0 < args.ewma_alpha <= 1.0:
        raise ValueError("ewma-alpha must be in (0, 1]")
    if args.cusum_drift < 0.0:
        raise ValueError("cusum-drift must be nonnegative")
    if not 0.5 < args.quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1)")
    if args.reservoir_size < 100:
        raise ValueError("reservoir-size must be at least 100")
    failures = verify_checksums(args.dataset) + verify_checksums(args.splits)
    if failures:
        raise ValueError("Checksum verification failed: " + "; ".join(failures))
    contract = load_contract()
    dataset_manifest = json.loads(
        (args.dataset / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if not (
        dataset_manifest.get("contract_version") == contract["contract_version"] == "2.2.0"
        and dataset_manifest.get("model_version") == contract["model_version"] == "2.2.0"
        and str(dataset_manifest.get("generator_version", "")).startswith("2.2.")
    ):
        raise ValueError(
            "Baseline requires model/contract/generator 2.2.x dataset"
        )
    signal_header = set(read_csv_header(args.dataset / "signals.csv"))
    input_forbidden = EXPLICIT_FORBIDDEN | (
        set(contract["always_forbidden_features"])
        - set(contract["signal_context_fields"])
    )
    forbidden = signal_header & input_forbidden
    if forbidden:
        raise ValueError(f"Refusing privileged baseline inputs: {sorted(forbidden)}")
    missing = set(BASELINE_INPUTS) - signal_header
    if missing:
        raise ValueError(f"Signals are missing baseline inputs: {sorted(missing)}")
    _prepare_output(args.output)

    episode_rows = list(read_csv_rows(args.dataset / "episodes.csv"))
    episode_meta = {row["episode_id"]: row for row in episode_rows}
    split_rows = list(read_csv_rows(args.splits / "episode_splits.csv"))
    episode_split = {row["episode_id"]: row["split"] for row in split_rows}
    if set(episode_meta) != set(episode_split):
        raise ValueError("Dataset episodes and split episodes differ")
    stats = _fit_stats(args.dataset, episode_meta, episode_split)
    thresholds, calibration = _calibrate(
        args.dataset,
        stats,
        contract["baseline_thresholds"],
        episode_split,
        episode_meta,
        args,
    )

    score_fields = [
        "episode_id",
        "time_start_s",
        "time_s",
        "split",
        "plant_mode",
        "fault_active",
        "safety_event",
        "unsafe_forward_l",
        *[field for method in METHODS for field in (f"{method}_score", f"{method}_alarm")],
    ]
    episode_metric_fields = [
        "episode_id",
        "split",
        "canonical_code",
        "method",
        "target_events",
        "detected_events",
        "first_detection_latency_s",
        "false_alarm_edges",
        "false_alarm_production_edges",
        "false_alarm_cip_edges",
        "negative_operational_seconds",
        "negative_production_seconds",
        "negative_cip_seconds",
        "false_alarm_seconds",
        "false_alarm_production_seconds",
        "false_alarm_cip_seconds",
        "unsafe_forward_l_before_detection",
        "point_tp",
        "point_fp",
        "point_fn",
        "point_tn",
    ]
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    score_count = 0
    episode_metric_count = 0
    score_path = args.output / "baseline_scores.csv"
    episode_metric_path = args.output / "episode_metrics.csv"
    with (
        score_path.open("w", newline="", encoding="utf-8") as score_handle,
        episode_metric_path.open("w", newline="", encoding="utf-8") as metric_handle,
    ):
        score_writer = write_csv_header(score_handle, score_fields)
        metric_writer = write_csv_header(metric_handle, episode_metric_fields)
        current_episode: str | None = None
        episode_buffer: list[tuple[dict[str, str], dict[str, str], dict[str, float]]] = []

        def flush_episode() -> None:
            nonlocal episode_metric_count
            if not episode_buffer:
                return
            episode_id = episode_buffer[0][0]["episode_id"]
            metrics, partial = _evaluate_episode(
                episode_buffer,
                episode_meta[episode_id],
                episode_split[episode_id],
                thresholds,
            )
            for metric in metrics:
                metric_writer.writerow(metric)
                episode_metric_count += 1
            _merge_aggregate(aggregate, partial)

        for signal, label, scores in _scored_rows(
            args.dataset,
            stats,
            contract["baseline_thresholds"],
            args.ewma_alpha,
            args.cusum_drift,
        ):
            episode = signal["episode_id"]
            if current_episode is not None and episode != current_episode:
                flush_episode()
                episode_buffer = []
            current_episode = episode
            episode_buffer.append((signal, label, scores))
            score_writer.writerow(
                {
                    "episode_id": episode,
                    "time_start_s": signal["time_start_s"],
                    "time_s": signal["time_s"],
                    "split": episode_split[episode],
                    "plant_mode": signal["plant_mode"],
                    "fault_active": label["fault_active"],
                    "safety_event": label["safety_event"],
                    "unsafe_forward_l": label["unsafe_forward_l"],
                    **{
                        f"{method}_score": scores[method]
                        for method in METHODS
                    },
                    **{
                        f"{method}_alarm": int(scores[method] >= thresholds[method])
                        for method in METHODS
                    },
                }
            )
            score_count += 1
        flush_episode()

    results_by_split: dict[str, dict[str, Any]] = defaultdict(dict)
    for (method, split), values in sorted(aggregate.items()):
        results_by_split[split][method] = _final_metrics(values)
    train_stats = {
        mode: {
            feature: {
                "count": item.count,
                "mean": item.mean,
                "std": item.std,
            }
            for feature, item in values.items()
        }
        for mode, values in sorted(stats.items())
    }
    results = {
        "baseline_version": BASELINE_VERSION,
        "dataset_manifest_sha256": sha256_file(args.dataset / "dataset_manifest.json"),
        "split_manifest_sha256": sha256_file(args.splits / "split_manifest.json"),
        "methods": list(METHODS),
        "inputs": list(BASELINE_INPUTS),
        "explicit_forbidden_inputs": sorted(input_forbidden),
        "parameters": {
            "ewma_alpha": args.ewma_alpha,
            "cusum_drift": args.cusum_drift,
            "calibration_quantile": args.quantile,
            "seed": args.seed,
        },
        "thresholds": thresholds,
        "calibration": calibration,
        "normal_behavior_codes": sorted(NORMAL_BEHAVIOR_CODES),
        "operational_gating": {
            "field": "cip_cycle_active",
            "plant_mode_used_for_scoring": False,
            "plant_mode_used_for_standardization": False,
            "plant_mode_used_for_state_reset": False,
        },
        "provenance": {
            "ml_contract.json": sha256_file(CONTRACT_PATH),
            "run_ml_baselines.py": sha256_file(Path(__file__).resolve()),
            "ml_pipeline_common.py": sha256_file(BASE_DIR / "ml_pipeline_common.py"),
        },
        "metrics_by_split": dict(sorted(results_by_split.items())),
        "counts": {
            "score_rows": score_count,
            "episode_metric_rows": episode_metric_count,
        },
        "training_statistics": train_stats,
        "limitations": [
            "These baselines are synthetic-data research references, not safety interlocks.",
            "Thresholds are calibrated without opening test splits; validation-ID N00/M01/M02 normal behaviors are preferred, with train-ID fallback for short smoke traces.",
            "CIP rule checks accept either caustic-like or acid-like pH response because the observable contract exposes concentration but not chemical identity.",
            "A detector alarm already active at event onset can count as detection and remains visible through false-alarm metrics.",
        ],
    }
    write_json(args.output / "baseline_results.json", results)
    manifest = {
        "baseline_version": BASELINE_VERSION,
        "dataset_manifest_sha256": results["dataset_manifest_sha256"],
        "split_manifest_sha256": results["split_manifest_sha256"],
        "artifacts": [
            "baseline_scores.csv",
            "episode_metrics.csv",
            "baseline_results.json",
            "baseline_manifest.json",
            "checksums.sha256",
        ],
    }
    write_json(args.output / "baseline_manifest.json", manifest)
    write_checksums(
        args.output,
        [
            "baseline_scores.csv",
            "episode_metrics.csv",
            "baseline_results.json",
            "baseline_manifest.json",
        ],
    )
    return args.output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    results = json.loads((output / "baseline_results.json").read_text(encoding="utf-8"))
    print(
        f"Evaluated {', '.join(results['methods'])} on {results['counts']['score_rows']} rows; "
        f"results={output / 'baseline_results.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
