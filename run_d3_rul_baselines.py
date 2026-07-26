#!/usr/bin/env python3
"""Run deterministic health ridge/slope and censor-aware D3-RUL baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    from .d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        feature_fields,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        read_csv_header,
        read_csv_rows,
        sha256_file,
        verify_checksums,
        write_checksums,
        write_csv_header,
        write_json,
    )
except ImportError:  # pragma: no cover
    from d3_rul_common import (
        BASE_DIR,
        CONTRACT_PATH,
        feature_fields,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        read_csv_header,
        read_csv_rows,
        sha256_file,
        verify_checksums,
        write_checksums,
        write_csv_header,
        write_json,
    )


BASELINE_VERSION = "1.0.0"
RAW_FEATURES = (
    "operating_time_meter_h",
    "steam_valve",
    "measured_differential_pressure_bar",
    "preheat_temp_sensor_c",
    "control_temp_sensor_c",
    "heater_power_sensor_kw",
    "pump_power_sensor_kw",
)
SLOPE_SOURCES = (
    "steam_valve",
    "measured_differential_pressure_bar",
    "heater_power_sensor_kw",
)
MODEL_FEATURES = (*RAW_FEATURES, *(f"{field}_slope_per_h" for field in SLOPE_SOURCES))


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
            return 1.0
        return max(1.0e-9, math.sqrt(max(0.0, self.m2 / (self.count - 1))))


class KaplanMeier:
    def __init__(self, outcomes: Sequence[tuple[float, bool]]) -> None:
        if not outcomes:
            raise ValueError("Kaplan-Meier needs at least one training trajectory")
        self.maximum_time = max(duration for duration, _ in outcomes)
        event_times = sorted({duration for duration, event in outcomes if event})
        survival = 1.0
        self.steps: list[tuple[float, float]] = []
        for time in event_times:
            at_risk = sum(duration + 1.0e-12 >= time for duration, _ in outcomes)
            events = sum(event and math.isclose(duration, time, abs_tol=1.0e-9) for duration, event in outcomes)
            if at_risk:
                survival *= 1.0 - events / at_risk
                self.steps.append((time, survival))

    def survival_at(self, time_s: float) -> float:
        value = 1.0
        for event_time, survival in self.steps:
            if event_time > time_s + 1.0e-12:
                break
            value = survival
        return value

    def median_remaining(self, age_s: float) -> float | None:
        base = self.survival_at(age_s)
        if base <= 0.0:
            return None
        target = 0.5 * base
        for event_time, survival in self.steps:
            if event_time + 1.0e-12 >= age_s and survival <= target:
                return max(0.0, event_time - age_s)
        # A censored follow-up endpoint is not a failure. If survival never
        # falls to half its conditional value, the KM median is unidentified.
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-lambda", type=float)
    parser.add_argument("--landmark-interval-s", type=float)
    return parser


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
            for field in ("trajectory_id", "time_start_sim_s", "time_sim_s"):
                if signal[field] != label[field]:
                    raise ValueError(f"signal/label mismatch at {field}")
            yield signal, label


def _feature_rows(
    dataset: Path, slope_window_h: float
) -> Iterator[tuple[dict[str, str], dict[str, str], dict[str, float]]]:
    previous_trajectory: str | None = None
    history: deque[tuple[float, dict[str, float]]] = deque()
    for signal, label in _joined_rows(dataset):
        trajectory = signal["trajectory_id"]
        if trajectory != previous_trajectory:
            history.clear()
            previous_trajectory = trajectory
        age_h = float(signal["operating_time_meter_h"])
        current = {field: float(signal[field]) for field in RAW_FEATURES}
        history.append((age_h, current))
        while len(history) > 1 and history[0][0] < age_h - slope_window_h:
            history.popleft()
        first_age, first = history[0]
        elapsed_h = age_h - first_age
        features = dict(current)
        for field in SLOPE_SOURCES:
            features[f"{field}_slope_per_h"] = (
                (current[field] - first[field]) / elapsed_h
                if elapsed_h > 1.0e-12
                else 0.0
            )
        yield signal, label, features


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-15:
            raise ValueError("ridge normal equations are singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[column])
                ]
    return [augmented[index][-1] for index in range(size)]


def _fit_ridge(
    dataset: Path,
    split_by_trajectory: Mapping[str, str],
    slope_window_h: float,
    ridge_lambda: float,
) -> tuple[dict[str, RunningStats], list[float], float]:
    stats = {feature: RunningStats() for feature in MODEL_FEATURES}
    health = RunningStats()
    for signal, label, features in _feature_rows(dataset, slope_window_h):
        if split_by_trajectory[signal["trajectory_id"]] != "train_id":
            continue
        for feature, value in features.items():
            stats[feature].add(value)
        health.add(float(label["health_index"]))
    if health.count == 0:
        raise ValueError("No train_id rows exist for D3 health fitting")
    dimension = len(MODEL_FEATURES) + 1
    matrix = [[0.0] * dimension for _ in range(dimension)]
    vector = [0.0] * dimension
    for signal, label, features in _feature_rows(dataset, slope_window_h):
        if split_by_trajectory[signal["trajectory_id"]] != "train_id":
            continue
        x = [1.0] + [
            (features[feature] - stats[feature].mean) / stats[feature].std
            for feature in MODEL_FEATURES
        ]
        target = float(label["health_index"])
        for row in range(dimension):
            vector[row] += x[row] * target
            for column in range(dimension):
                matrix[row][column] += x[row] * x[column]
    for index in range(1, dimension):
        matrix[index][index] += ridge_lambda
    return stats, _solve(matrix, vector), health.mean


def _ridge_predict(
    features: Mapping[str, float], stats: Mapping[str, RunningStats], coefficients: Sequence[float]
) -> float:
    value = coefficients[0]
    for coefficient, feature in zip(coefficients[1:], MODEL_FEATURES):
        value += coefficient * (
            (features[feature] - stats[feature].mean) / stats[feature].std
        )
    return min(1.0, max(0.0, value))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1) + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(truth: Sequence[float], prediction: Sequence[float]) -> float | None:
    if len(truth) < 2:
        return None
    left, right = _ranks(truth), _ranks(prediction)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _health_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    return {
        "rows": len(truth),
        "mae": statistics.fmean(abs(x - y) for x, y in zip(truth, prediction))
        if truth
        else None,
        "spearman": _spearman(truth, prediction),
    }


def _concordance(
    outcomes: Sequence[tuple[float, bool, float]]
) -> float | None:
    comparable = 0
    score = 0.0
    for left_index, left in enumerate(outcomes):
        for right in outcomes[left_index + 1 :]:
            left_time, left_event, left_prediction = left
            right_time, right_event, right_prediction = right
            if left_event and left_time < right_time - 1.0e-9:
                comparable += 1
                score += 1.0 if left_prediction < right_prediction else 0.5 if left_prediction == right_prediction else 0.0
            elif right_event and right_time < left_time - 1.0e-9:
                comparable += 1
                score += 1.0 if right_prediction < left_prediction else 0.5 if left_prediction == right_prediction else 0.0
    return score / comparable if comparable else None


def run(args: argparse.Namespace) -> Path:
    contract = load_d3_contract()
    failures = verify_checksums(args.dataset) + verify_checksums(args.splits)
    if failures:
        raise ValueError("Checksum verification failed: " + "; ".join(failures))
    manifest_path = args.dataset / "dataset_manifest.json"
    split_manifest_path = args.splits / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("dataset_family") == split_manifest.get("dataset_family") == "D3-RUL"
        and manifest.get("contract_version") == contract["contract_version"]
        and model_version_is_compatible(str(manifest.get("model_version", "")), contract)
        and split_manifest.get("dataset_manifest_sha256") == sha256_file(manifest_path)
    ):
        raise ValueError("Baselines require a compatible, bound D3-RUL dataset/split")
    header = set(read_csv_header(args.dataset / "signals.csv"))
    expected = set(feature_fields(contract)) | set(contract["signal_context_fields"])
    forbidden = set(contract["always_forbidden_signal_fields"])
    if header != expected or header & forbidden:
        raise ValueError(
            f"Refusing invalid D3 signals: unknown={sorted(header - expected)}, "
            f"missing={sorted(expected - header)}, forbidden={sorted(header & forbidden)}"
        )
    split_rows = list(read_csv_rows(args.splits / "trajectory_splits.csv"))
    trajectories = list(read_csv_rows(args.dataset / "trajectories.csv"))
    trajectory_by_id = {row["trajectory_id"]: row for row in trajectories}
    split_ids = [row["trajectory_id"] for row in split_rows]
    split_by_trajectory = {row["trajectory_id"]: row["split"] for row in split_rows}
    profile_owners: dict[str, str] = {}
    split_metadata_valid = (
        len(split_ids) == len(set(split_ids))
        and set(split_by_trajectory) == set(trajectory_by_id)
        and split_manifest.get("policy", {}).get("outcomes_used_for_assignment") is False
    )
    for row in split_rows:
        trajectory = trajectory_by_id.get(row["trajectory_id"])
        if trajectory is None:
            split_metadata_valid = False
            continue
        for field in (
            "life_family_id",
            "plant_profile_id",
            "profile_config_hash",
            "domain",
            "noise_seed",
            "degradation_seed",
            "censor_seed",
        ):
            split_metadata_valid = split_metadata_valid and row[field] == trajectory[field]
        profile = row["plant_profile_id"]
        previous = profile_owners.setdefault(profile, row["split"])
        split_metadata_valid = split_metadata_valid and previous == row["split"]
    if not split_metadata_valid:
        raise ValueError(
            "Malformed D3 split: IDs/metadata must match trajectories and every profile "
            "must remain in one outcome-independent split"
        )
    prepare_empty_output(args.output)
    baseline_contract = contract["baseline"]
    ridge_lambda = float(
        baseline_contract["ridge_lambda"]
        if args.ridge_lambda is None
        else args.ridge_lambda
    )
    landmark_interval_s = float(
        baseline_contract["landmark_interval_sim_s"]
        if args.landmark_interval_s is None
        else args.landmark_interval_s
    )
    slope_window_h = float(baseline_contract["slope_window_sim_h"])
    if (
        not math.isfinite(ridge_lambda)
        or ridge_lambda < 0.0
        or not math.isfinite(landmark_interval_s)
        or landmark_interval_s <= 0.0
        or not math.isfinite(slope_window_h)
        or slope_window_h <= 0.0
    ):
        raise ValueError("ridge-lambda must be nonnegative and landmark interval positive")
    stats, coefficients, constant_health = _fit_ridge(
        args.dataset,
        split_by_trajectory,
        slope_window_h,
        ridge_lambda,
    )
    train_outcomes = [
        (
            float(row["observation_end_sim_s"]),
            row["event_observed"] == "1",
        )
        for row in trajectories
        if split_by_trajectory[row["trajectory_id"]] == "train_id"
    ]
    km = KaplanMeier(train_outcomes)
    observed_train_ages = [duration for duration, event in train_outcomes if event]
    if not observed_train_ages:
        raise ValueError(
            "D3 RUL baselines require at least one observed EOL in train_id; "
            "administrative censor times are never used as pseudo-failures"
        )
    naive_event_age = statistics.median(observed_train_ages)

    health_fields = [
        "trajectory_id",
        "time_sim_s",
        "split",
        "health_index",
        "constant_prediction",
        "ridge_prediction",
    ]
    rul_fields = [
        "trajectory_id",
        "time_sim_s",
        "operating_time_sim_s",
        "split",
        "event_observed",
        "time_to_event_or_censor_sim_s",
        "true_rul_sim_s",
        "naive_rul_sim_s",
        "kaplan_meier_rul_sim_s",
        "ridge_slope_rul_sim_s",
    ]
    health_values: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    rul_errors: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    first_landmark: dict[tuple[str, str], tuple[float, bool, float]] = {}
    prediction_history: deque[tuple[float, float]] = deque()
    current_trajectory: str | None = None
    next_landmark_s = 0.0
    health_count = rul_count = 0
    health_path = args.output / "health_predictions.csv"
    rul_path = args.output / "rul_predictions.csv"
    with (
        health_path.open("w", newline="", encoding="utf-8") as health_handle,
        rul_path.open("w", newline="", encoding="utf-8") as rul_handle,
    ):
        health_writer = write_csv_header(health_handle, health_fields)
        rul_writer = write_csv_header(rul_handle, rul_fields)
        for signal, label, features in _feature_rows(args.dataset, slope_window_h):
            trajectory_id = signal["trajectory_id"]
            if trajectory_id != current_trajectory:
                current_trajectory = trajectory_id
                prediction_history.clear()
                next_landmark_s = 0.0
            split = split_by_trajectory[trajectory_id]
            truth_health = float(label["health_index"])
            ridge_health = _ridge_predict(features, stats, coefficients)
            health_writer.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "time_sim_s": signal["time_sim_s"],
                    "split": split,
                    "health_index": truth_health,
                    "constant_prediction": constant_health,
                    "ridge_prediction": ridge_health,
                }
            )
            for method, prediction in (
                ("constant", constant_health),
                ("ridge", ridge_health),
            ):
                truth_values, predictions = health_values[(split, method)]
                truth_values.append(truth_health)
                predictions.append(prediction)
            health_count += 1

            age_h = float(signal["operating_time_meter_h"])
            prediction_history.append((age_h, ridge_health))
            while (
                len(prediction_history) > 1
                and prediction_history[0][0] < age_h - slope_window_h
            ):
                prediction_history.popleft()
            time_sim_s = float(signal["time_sim_s"])
            if time_sim_s + 1.0e-9 < next_landmark_s:
                continue
            while next_landmark_s <= time_sim_s + 1.0e-9:
                next_landmark_s += landmark_interval_s
            age_s = age_h * 3600.0
            naive_rul = max(0.0, naive_event_age - age_s)
            km_rul = km.median_remaining(age_s)
            first_age, first_health = prediction_history[0]
            elapsed_h = age_h - first_age
            health_slope = (
                (ridge_health - first_health) / elapsed_h
                if elapsed_h > 1.0e-12
                else 0.0
            )
            eol_health = 1.0 - float(contract["eol"]["fouling_index_threshold"])
            if health_slope < -float(baseline_contract["minimum_slope_per_h"]):
                slope_rul = max(0.0, (ridge_health - eol_health) / -health_slope * 3600.0)
                slope_rul = min(slope_rul, 2.0 * km.maximum_time)
            else:
                slope_rul = km_rul if km_rul is not None else naive_rul
            true_rul = label["rul_sim_s"]
            duration = float(label["time_to_event_or_censor_sim_s"])
            event = label["event_observed"] == "1"
            rul_writer.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "time_sim_s": signal["time_sim_s"],
                    "operating_time_sim_s": age_s,
                    "split": split,
                    "event_observed": int(event),
                    "time_to_event_or_censor_sim_s": duration,
                    "true_rul_sim_s": true_rul,
                    "naive_rul_sim_s": naive_rul,
                    "kaplan_meier_rul_sim_s": "" if km_rul is None else km_rul,
                    "ridge_slope_rul_sim_s": slope_rul,
                }
            )
            predictions_by_method: dict[str, float | None] = {
                "naive": naive_rul,
                "kaplan_meier": km_rul,
                "ridge_slope": slope_rul,
            }
            outcome_age = float(
                trajectory_by_id[trajectory_id]["observation_end_sim_s"]
            )
            for method, prediction in predictions_by_method.items():
                if prediction is None:
                    continue
                first_landmark.setdefault(
                    (trajectory_id, method),
                    (outcome_age, event, age_s + prediction),
                )
                if event and true_rul:
                    rul_errors[(split, method)].append(
                        (float(true_rul), prediction, outcome_age)
                    )
            rul_count += 1

    metrics_by_split: dict[str, Any] = {}
    for split in sorted(set(split_by_trajectory.values())):
        health_metrics = {
            method: _health_metrics(*health_values[(split, method)])
            for method in ("constant", "ridge")
        }
        rul_metrics: dict[str, Any] = {}
        for method in ("naive", "kaplan_meier", "ridge_slope"):
            errors = rul_errors[(split, method)]
            absolute = [abs(truth - prediction) for truth, prediction, _ in errors]
            normalized = [
                abs(truth - prediction) / full_life
                for truth, prediction, full_life in errors
                if full_life > 0.0
            ]
            concordance_rows = [
                first_landmark[(trajectory["trajectory_id"], method)]
                for trajectory in trajectories
                if split_by_trajectory[trajectory["trajectory_id"]] == split
                and (trajectory["trajectory_id"], method) in first_landmark
            ]
            rul_metrics[method] = {
                "observed_event_landmarks": len(errors),
                "mae_s": statistics.fmean(absolute) if absolute else None,
                "normalized_mae": statistics.fmean(normalized) if normalized else None,
                "within_20_percent": statistics.fmean(
                    abs(truth - prediction) <= 0.20 * max(truth, 1.0)
                    for truth, prediction, _ in errors
                )
                if errors
                else None,
                "trajectory_level_c_index": _concordance(concordance_rows),
                "censor_note": "Exact RUL errors use observed EOL rows only; censored trajectories enter the trajectory-level concordance calculation without RUL imputation.",
            }
        metrics_by_split[split] = {"health": health_metrics, "rul": rul_metrics}

    results = {
        "dataset_family": "D3-RUL",
        "baseline_version": BASELINE_VERSION,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "fit_population": "train_id only",
        "methods": {
            "health": ["constant", "ridge"],
            "rul": ["naive", "kaplan_meier", "ridge_slope"],
        },
        "inputs": list(MODEL_FEATURES),
        "parameters": {
            "ridge_lambda": ridge_lambda,
            "slope_window_sim_h": slope_window_h,
            "landmark_interval_sim_s": landmark_interval_s,
            "naive_train_event_age_sim_s": naive_event_age,
        },
        "ridge": {
            "coefficients": coefficients,
            "training_feature_statistics": {
                feature: {
                    "count": stats[feature].count,
                    "mean": stats[feature].mean,
                    "std": stats[feature].std,
                }
                for feature in MODEL_FEATURES
            },
        },
        "kaplan_meier": {
            "training_trajectories": len(train_outcomes),
            "training_events": sum(event for _, event in train_outcomes),
            "steps": [
                {"time_sim_s": time, "survival": survival}
                for time, survival in km.steps
            ],
        },
        "metrics_by_split": metrics_by_split,
        "counts": {
            "health_prediction_rows": health_count,
            "rul_landmark_rows": rul_count,
        },
        "limitations": [
            "These are scalar-fouling synthetic research baselines, not a maintenance release criterion.",
            "Kaplan-Meier is unconditional; ridge-slope extrapolates a supervised health estimate and falls back to KM when its historical slope is nonnegative.",
            "Censored exact RUL remains null and is never replaced with an episode-end pseudo-failure.",
        ],
        "provenance": {
            "run_d3_rul_baselines.py": sha256_file(Path(__file__).resolve()),
            "d3_rul_common.py": sha256_file(BASE_DIR / "d3_rul_common.py"),
            "d3_rul_contract.json": sha256_file(CONTRACT_PATH),
        },
    }
    write_json(args.output / "baseline_results.json", results)
    baseline_manifest = {
        "dataset_family": "D3-RUL",
        "baseline_version": BASELINE_VERSION,
        "dataset_manifest_sha256": results["dataset_manifest_sha256"],
        "split_manifest_sha256": results["split_manifest_sha256"],
        "artifacts": [
            "health_predictions.csv",
            "rul_predictions.csv",
            "baseline_results.json",
            "baseline_manifest.json",
            "checksums.sha256",
        ],
    }
    write_json(args.output / "baseline_manifest.json", baseline_manifest)
    write_checksums(
        args.output,
        [
            "health_predictions.csv",
            "rul_predictions.csv",
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
    print(f"Wrote D3-RUL baselines: {(output / 'baseline_results.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
