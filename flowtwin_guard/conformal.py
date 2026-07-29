"""Finite-sample conformal prediction and fail-closed abstention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CONFORMAL_VERSION = "0.2.0"
DECISIONS = frozenset({"DIAGNOSE", "REVIEW", "UNKNOWN"})


def corrected_quantile(values: Sequence[float], alpha: float) -> float:
    """Return split-conformal ``ceil((n+1)(1-alpha))`` order statistic."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("conformal scores must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("conformal scores must be finite")
    rank = min(array.size, int(math.ceil((array.size + 1) * (1.0 - alpha))))
    return float(np.partition(array, rank - 1)[rank - 1])


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C] with C>=2")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")


def _mode_name(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "cip" if bool(value) else "production"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("mode relay must be finite")
        return "cip" if number >= 0.5 else "production"
    text = str(value).strip().lower()
    if text not in {"cip", "production"}:
        raise ValueError(f"unsupported calibration mode {value!r}")
    return text


def counterfactual_group_mode_block_max(
    probabilities: np.ndarray,
    targets: np.ndarray,
    modes: Sequence[object],
    energies: np.ndarray,
    groups: Sequence[object],
    profiles: Sequence[object],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create conservative calibration items from correlated simulator rows.

    A counterfactual group shares a profile, replicate and disturbance stream.
    For each group and observable process mode, the returned class item is the
    row with maximum true-class nonconformity and its energy is replaced by the
    maximum ID energy in that block.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    energies = np.asarray(energies, dtype=np.float64)
    _validate_probabilities(probabilities)
    n, classes = probabilities.shape
    if targets.shape != (n,) or energies.shape != (n,):
        raise ValueError("block calibration arrays have inconsistent lengths")
    if len(modes) != n or len(groups) != n or len(profiles) != n:
        raise ValueError("block calibration metadata have inconsistent lengths")
    if np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("block calibration target is outside class range")
    if not np.isfinite(energies).all():
        raise ValueError("block calibration energy scores must be finite")
    mode_names = np.asarray([_mode_name(value) for value in modes], dtype=object)
    group_names = np.asarray([str(value) for value in groups], dtype=object)
    profile_names = np.asarray([str(value) for value in profiles], dtype=object)
    scores = 1.0 - probabilities[np.arange(n), targets]
    selected_probabilities: list[np.ndarray] = []
    selected_targets: list[int] = []
    selected_modes: list[str] = []
    selected_energies: list[float] = []
    block_profiles: list[str] = []
    for group in sorted(set(group_names.tolist())):
        group_rows = group_names == group
        unique_profiles = np.unique(profile_names[group_rows])
        if unique_profiles.size != 1:
            raise ValueError("counterfactual calibration group spans plant profiles")
        for mode in ("production", "cip"):
            indexes = np.flatnonzero(group_rows & (mode_names == mode))
            if not indexes.size:
                continue
            worst = int(indexes[np.argmax(scores[indexes])])
            selected_probabilities.append(probabilities[worst])
            selected_targets.append(int(targets[worst]))
            selected_modes.append(mode)
            selected_energies.append(float(np.max(energies[indexes])))
            block_profiles.append(str(unique_profiles[0]))
    if not selected_probabilities:
        raise ValueError("validation set contains no counterfactual group/mode blocks")
    arrays = {
        "probabilities": np.asarray(selected_probabilities, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "modes": np.asarray(selected_modes, dtype=np.str_),
        "energies": np.asarray(selected_energies, dtype=np.float64),
    }
    audit = {
        "method": "counterfactual_group_by_observable_mode_block_max",
        "source_rows": int(n),
        "calibration_blocks": int(len(selected_targets)),
        "counterfactual_groups": int(np.unique(group_names).size),
        "plant_profiles": int(np.unique(block_profiles).size),
        "class_score": "maximum true-class nonconformity within each block",
        "energy_score": "maximum ID energy within each block",
        "guarantee_boundary": (
            "empirical synthetic row coverage only; independent validation "
            "profiles are required for a profile-shift finite-sample guarantee"
        ),
    }
    return arrays, audit


@dataclass(frozen=True)
class ConformalResult:
    prediction_set: tuple[int, ...]
    decision: str


@dataclass
class ModeConformalCalibrator:
    alpha: float = 0.10
    ood_alpha: float = 0.01
    minimum_mode_samples: int = 20
    global_threshold: float | None = None
    mode_thresholds: dict[str, float] | None = None
    energy_threshold: float | None = None
    calibration_count: int = 0

    def fit(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        modes: Sequence[object],
        energies: np.ndarray,
        *,
        splits: Sequence[str],
    ) -> "ModeConformalCalibrator":
        probabilities = np.asarray(probabilities, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        energies = np.asarray(energies, dtype=np.float64)
        _validate_probabilities(probabilities)
        n, classes = probabilities.shape
        if targets.shape != (n,) or energies.shape != (n,) or len(modes) != n:
            raise ValueError("calibration arrays have inconsistent lengths")
        if len(splits) != n or set(map(str, splits)) != {"validation_id"}:
            raise ValueError("conformal calibration may use validation_id only")
        if not np.isfinite(energies).all():
            raise ValueError("calibration energy scores must be finite")
        if np.any(targets < 0) or np.any(targets >= classes):
            raise ValueError("calibration target is outside class range")
        mode_names = np.asarray([_mode_name(value) for value in modes], dtype=object)
        scores = 1.0 - probabilities[np.arange(n), targets]
        self.global_threshold = corrected_quantile(scores, self.alpha)
        self.mode_thresholds = {}
        for mode in ("production", "cip"):
            selected = scores[mode_names == mode]
            if selected.size >= self.minimum_mode_samples:
                self.mode_thresholds[mode] = corrected_quantile(selected, self.alpha)
        self.energy_threshold = corrected_quantile(energies, self.ood_alpha)
        self.calibration_count = n
        return self

    def _require_fit(self) -> None:
        if (
            self.global_threshold is None
            or self.mode_thresholds is None
            or self.energy_threshold is None
            or self.calibration_count <= 0
        ):
            raise ValueError("conformal calibrator is not fitted")

    def predict(
        self, probability: Sequence[float], mode: object, energy: float
    ) -> ConformalResult:
        self._require_fit()
        row = np.asarray(probability, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("probability must be a vector")
        _validate_probabilities(row.reshape(1, -1))
        energy = float(energy)
        if not math.isfinite(energy):
            raise ValueError("energy must be finite")
        mode_name = _mode_name(mode)
        threshold = self.mode_thresholds.get(mode_name, self.global_threshold)
        assert threshold is not None
        prediction_set = tuple(
            int(index) for index, value in enumerate(row) if 1.0 - value <= threshold
        )
        if energy > float(self.energy_threshold):
            decision = "UNKNOWN"
        elif len(prediction_set) == 0:
            decision = "UNKNOWN"
        elif len(prediction_set) == 1:
            decision = "DIAGNOSE"
        else:
            decision = "REVIEW"
        return ConformalResult(prediction_set=prediction_set, decision=decision)

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "version": CONFORMAL_VERSION,
            "method": "mode_conditioned_split_conformal",
            "alpha": self.alpha,
            "ood_alpha": self.ood_alpha,
            "minimum_mode_samples": self.minimum_mode_samples,
            "global_threshold": self.global_threshold,
            "mode_thresholds": dict(sorted((self.mode_thresholds or {}).items())),
            "energy_threshold": self.energy_threshold,
            "calibration_count": self.calibration_count,
            "calibration_split": "validation_id",
            "decision_vocabulary": sorted(DECISIONS),
            "safety_claim": "No prediction authorizes SAFE or RELEASE.",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModeConformalCalibrator":
        if value.get("version") != CONFORMAL_VERSION:
            raise ValueError("unsupported conformal state version")
        calibrator = cls(
            alpha=float(value["alpha"]),
            ood_alpha=float(value["ood_alpha"]),
            minimum_mode_samples=int(value["minimum_mode_samples"]),
        )
        calibrator.global_threshold = float(value["global_threshold"])
        calibrator.mode_thresholds = {
            str(name): float(threshold)
            for name, threshold in dict(value["mode_thresholds"]).items()
        }
        calibrator.energy_threshold = float(value["energy_threshold"])
        calibrator.calibration_count = int(value["calibration_count"])
        calibrator._require_fit()
        return calibrator


__all__ = [
    "CONFORMAL_VERSION",
    "DECISIONS",
    "ConformalResult",
    "ModeConformalCalibrator",
    "counterfactual_group_mode_block_max",
    "corrected_quantile",
]
