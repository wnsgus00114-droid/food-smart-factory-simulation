"""Streaming, profile-disjoint window loader for FlowTwin-Guard.

The loader joins ``signals.csv`` and ``oracle_labels.csv`` only by their three
declared keys.  Oracle fields are converted to target tensors and are never
placed in the feature or route-control tensors.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ml_pipeline_common import BASE_DIR, load_contract, read_csv_rows

from .graph import CONTROL_FIELDS, ProcessGraph


DATA_LOADER_VERSION = "0.4.1"
NORMAL_BEHAVIOR_CODES = frozenset({"N00", "M01", "M02"})
JOIN_KEYS = ("episode_id", "time_start_s", "time_s")


@dataclass(frozen=True)
class TargetContract:
    codes: tuple[str, ...]
    locations: tuple[str, ...]
    mechanisms: tuple[str, ...]
    code_to_index: Mapping[str, int]
    code_to_location: tuple[int, ...]
    code_to_mechanism: tuple[int, ...]
    code_is_fault: tuple[bool, ...]

    @property
    def normal_index(self) -> int:
        return self.code_to_index["N00"]


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def load_target_contract(
    path: Path | None = None,
    *,
    ml_contract_path: Path | None = None,
    graph: ProcessGraph | None = None,
) -> TargetContract:
    source = path or BASE_DIR / "fault_asset_contract.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0.0":
        raise ValueError("unsupported fault_asset_contract schema")
    targets = value.get("targets")
    if not isinstance(targets, list):
        raise ValueError("fault_asset_contract.targets must be an array")
    ml_contract = load_contract(ml_contract_path)
    canonical = tuple(str(item["code"]) for item in ml_contract["taxonomy"])
    codes = tuple(str(item["code"]) for item in targets)
    if codes != canonical:
        raise ValueError(
            "fault target order must exactly match ml_contract taxonomy: "
            f"expected {canonical}, got {codes}"
        )
    locations = _ordered_unique([str(item["location"]) for item in targets])
    mechanisms = _ordered_unique([str(item["mechanism"]) for item in targets])
    special_locations = set(map(str, value.get("special_locations", [])))
    if graph is not None:
        invalid = set(locations) - set(graph.node_tags) - special_locations
        if invalid:
            raise ValueError(f"fault target locations are not P&ID assets: {sorted(invalid)}")
    location_index = {name: index for index, name in enumerate(locations)}
    mechanism_index = {name: index for index, name in enumerate(mechanisms)}
    fault_by_code = {
        str(item["code"]): bool(item["is_fault"])
        for item in ml_contract["taxonomy"]
    }
    return TargetContract(
        codes=codes,
        locations=locations,
        mechanisms=mechanisms,
        code_to_index={code: index for index, code in enumerate(codes)},
        code_to_location=tuple(location_index[str(item["location"])] for item in targets),
        code_to_mechanism=tuple(
            mechanism_index[str(item["mechanism"])] for item in targets
        ),
        code_is_fault=tuple(fault_by_code[code] for code in codes),
    )


@dataclass
class FeatureStandardizer:
    features: tuple[str, ...]
    count: int
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        dataset: Path,
        splits: Path,
        graph: ProcessGraph,
        *,
        accepted_splits: Sequence[str] = ("train_id",),
    ) -> "FeatureStandardizer":
        split_map = {
            row["episode_id"]: row["split"]
            for row in read_csv_rows(splits / "episode_splits.csv")
        }
        accepted = set(accepted_splits)
        count = 0
        mean = np.zeros(len(graph.features), dtype=np.float64)
        m2 = np.zeros(len(graph.features), dtype=np.float64)
        with (dataset / "signals.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _validate_signal_header(reader.fieldnames or [], graph)
            for row in reader:
                if split_map.get(row["episode_id"]) not in accepted:
                    continue
                values = np.asarray([float(row[name]) for name in graph.features])
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite observable in {row['episode_id']}")
                count += 1
                delta = values - mean
                mean += delta / count
                m2 += delta * (values - mean)
        if count < 2:
            raise ValueError("at least two training rows are required to fit scaling")
        scale = np.sqrt(np.maximum(m2 / (count - 1), 0.0))
        scale = np.maximum(scale, 1e-6)
        return cls(tuple(graph.features), count, mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.features):
            raise ValueError("feature width does not match fitted standardizer")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DATA_LOADER_VERSION,
            "fit_split": "train_id",
            "features": list(self.features),
            "count": self.count,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureStandardizer":
        features = tuple(map(str, value["features"]))
        mean = np.asarray(value["mean"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
        if mean.shape != scale.shape or mean.shape != (len(features),):
            raise ValueError("invalid standardizer vector shape")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("standardizer contains non-finite values")
        if np.any(scale <= 0.0):
            raise ValueError("standardizer scales must be positive")
        return cls(features, int(value["count"]), mean, scale)


def _validate_signal_header(fieldnames: Sequence[str], graph: ProcessGraph) -> None:
    contract = load_contract()
    header = set(fieldnames)
    missing = set(graph.features) - header
    if missing:
        raise ValueError(f"signals.csv lacks graph features: {sorted(missing)}")
    # Join/sequence context may coexist in the file but cannot enter graph.features.
    context = set(contract["signal_context_fields"])
    forbidden_present = (set(contract["always_forbidden_features"]) - context) & header
    if forbidden_present:
        raise ValueError(
            "oracle/forbidden fields entered signals.csv: "
            f"{sorted(forbidden_present)}"
        )


@dataclass
class _Episode:
    episode_id: str
    split: str
    meta: Mapping[str, str]
    profile: Mapping[str, str]
    x_raw: np.ndarray
    controls_raw: np.ndarray
    time_s: np.ndarray
    dt_s: np.ndarray
    class_target: np.ndarray
    anomaly_target: np.ndarray
    location_target: np.ndarray
    mechanism_target: np.ndarray
    behavior_active: np.ndarray
    fault_active: np.ndarray
    safety_event: np.ndarray
    unsafe_forward_l: np.ndarray


def _float_or_nan(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _read_metadata(dataset: Path, splits: Path) -> tuple[
    dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, str]
]:
    episodes = {row["episode_id"]: row for row in read_csv_rows(dataset / "episodes.csv")}
    profiles = {
        row["plant_profile_id"]: row for row in read_csv_rows(dataset / "profiles.csv")
    }
    split_rows = list(read_csv_rows(splits / "episode_splits.csv"))
    split_map = {row["episode_id"]: row["split"] for row in split_rows}
    if set(episodes) != set(split_map):
        raise ValueError("episode metadata and split assignments differ")
    group_splits: dict[str, set[str]] = {}
    for episode_id, meta in episodes.items():
        group_splits.setdefault(meta["counterfactual_group_id"], set()).add(
            split_map[episode_id]
        )
    leaked_groups = [group for group, values in group_splits.items() if len(values) != 1]
    if leaked_groups:
        raise ValueError(f"counterfactual groups cross splits: {leaked_groups[:5]}")
    return episodes, profiles, split_map


def _joint_episode_rows(
    dataset: Path,
    *,
    accepted_episode_ids: Collection[str] | None = None,
) -> Iterator[tuple[str, list[dict[str, str]], list[dict[str, str]]]]:
    """Stream aligned episode rows, retaining only pre-authorized episodes.

    The split firewall is evaluated from metadata before this function is
    entered.  Rejected episodes must still be streamed past in the monolithic
    CSV files, but their signal/oracle dictionaries are never accumulated into
    episode buffers and can therefore never reach ``_materialize_episode``.
    ``None`` preserves the all-episode behavior used by cache construction.
    """

    accepted = (
        None
        if accepted_episode_ids is None
        else frozenset(map(str, accepted_episode_ids))
    )
    with (
        (dataset / "signals.csv").open(newline="", encoding="utf-8") as signal_handle,
        (dataset / "oracle_labels.csv").open(newline="", encoding="utf-8") as label_handle,
    ):
        signals = csv.DictReader(signal_handle)
        labels = csv.DictReader(label_handle)
        current_id: str | None = None
        retain_current = False
        signal_rows: list[dict[str, str]] = []
        label_rows: list[dict[str, str]] = []
        while True:
            signal = next(signals, None)
            label = next(labels, None)
            if signal is None or label is None:
                if signal is not None or label is not None:
                    raise ValueError("signals.csv and oracle_labels.csv have different lengths")
                if current_id is not None and retain_current:
                    yield current_id, signal_rows, label_rows
                return
            for key in JOIN_KEYS:
                if signal[key] != label[key]:
                    raise ValueError(f"signal/oracle join mismatch at {key}")
            episode_id = signal["episode_id"]
            if current_id is None:
                current_id = episode_id
                retain_current = accepted is None or episode_id in accepted
            if episode_id != current_id:
                if retain_current:
                    yield current_id, signal_rows, label_rows
                current_id = episode_id
                retain_current = accepted is None or episode_id in accepted
                signal_rows = []
                label_rows = []
            if retain_current:
                signal_rows.append(signal)
                label_rows.append(label)


def _materialize_episode(
    episode_id: str,
    signals: Sequence[Mapping[str, str]],
    labels: Sequence[Mapping[str, str]],
    *,
    graph: ProcessGraph,
    targets: TargetContract,
    episodes: Mapping[str, Mapping[str, str]],
    profiles: Mapping[str, Mapping[str, str]],
    split_map: Mapping[str, str],
) -> _Episode:
    meta = episodes[episode_id]
    profile = profiles[meta["plant_profile_id"]]
    n = len(signals)
    x = np.empty((n, len(graph.features)), dtype=np.float32)
    controls = np.empty((n, len(CONTROL_FIELDS)), dtype=np.float32)
    time_s = np.empty(n, dtype=np.float32)
    dt_s = np.empty(n, dtype=np.float32)
    class_target = np.empty(n, dtype=np.int64)
    anomaly_target = np.empty(n, dtype=np.int64)
    location_target = np.empty(n, dtype=np.int64)
    mechanism_target = np.empty(n, dtype=np.int64)
    behavior_active = np.empty(n, dtype=np.float32)
    fault_active = np.empty(n, dtype=np.float32)
    safety_event = np.empty(n, dtype=np.float32)
    unsafe_forward_l = np.empty(n, dtype=np.float32)
    episode_code = str(meta["canonical_code"])
    episode_code_index = targets.code_to_index[episode_code]
    normal = targets.normal_index
    effect_time = _float_or_nan(meta.get("behavior_effect_time_s", ""))
    detection_eligible = bool(int(meta.get("detection_eligible", "0")))

    for index, (signal, label) in enumerate(zip(signals, labels, strict=True)):
        x[index] = [float(signal[name]) for name in graph.features]
        controls[index] = [float(signal[name]) for name in CONTROL_FIELDS]
        time_s[index] = float(signal["time_start_s"])
        dt_s[index] = float(signal["step_dt_s"])
        active = bool(int(label["behavior_active"]))
        visible = (
            episode_code != "N00"
            and detection_eligible
            and active
            and math.isfinite(effect_time)
            and time_s[index] + 1e-9 >= effect_time
        )
        target_index = episode_code_index if visible else normal
        class_target[index] = target_index
        anomaly_target[index] = int(targets.code_is_fault[target_index])
        location_target[index] = targets.code_to_location[target_index]
        mechanism_target[index] = targets.code_to_mechanism[target_index]
        behavior_active[index] = float(active)
        fault_active[index] = float(label["fault_active"])
        safety_event[index] = float(label["safety_event"])
        unsafe_forward_l[index] = float(label["unsafe_forward_l"])
    for name, array in {
        "features": x,
        "controls": controls,
        "time": time_s,
        "dt": dt_s,
        "behavior_active": behavior_active,
        "fault_active": fault_active,
        "safety_event": safety_event,
        "unsafe_forward_l": unsafe_forward_l,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values in {episode_id}")
    return _Episode(
        episode_id=episode_id,
        split=split_map[episode_id],
        meta=meta,
        profile=profile,
        x_raw=x,
        controls_raw=controls,
        time_s=time_s,
        dt_s=dt_s,
        class_target=class_target,
        anomaly_target=anomaly_target,
        location_target=location_target,
        mechanism_target=mechanism_target,
        behavior_active=behavior_active,
        fault_active=fault_active,
        safety_event=safety_event,
        unsafe_forward_l=unsafe_forward_l,
    )


def _window_ranges(length: int, window: int, stride: int) -> Iterator[tuple[int, int, int]]:
    """Yield (start, end, new_start) with each row evaluated exactly once."""

    if length <= 0:
        return
    previous_end = 0
    while previous_end < length:
        start = 0 if previous_end == 0 else max(0, previous_end - (window - stride))
        end = min(length, start + window)
        new_start = max(0, previous_end - start)
        if end <= previous_end:
            raise RuntimeError("window progression stalled")
        yield start, end, new_start
        previous_end = end


def _sample_window_ranges(
    ranges: Sequence[tuple[int, int, int]],
    maximum: int,
    *,
    priority_positions: Sequence[int] = (),
) -> tuple[tuple[int, int, int], ...]:
    """Select deterministic event-aware, timeline-spanning episode windows.

    The cap is deliberately per episode rather than global: every training
    episode remains represented and file ordering cannot silently determine
    which profiles/classes reach the optimiser.  If train-only target positions
    are supplied, onset/middle/end landmarks are selected in the non-overlapping
    loss/evaluation region of a window before the remaining slots are spread
    over the full timeline.  Evaluation never uses this cap.
    """

    values = tuple(ranges)
    if maximum <= 0 or len(values) <= maximum:
        return values

    positions = sorted({int(position) for position in priority_positions})
    if positions and (positions[0] < 0 or positions[-1] >= values[-1][1]):
        raise ValueError("priority positions must lie inside the episode")

    priority_indexes: list[int] = []
    if positions:
        landmark_count = min(3, maximum, len(positions))
        if landmark_count == 1:
            landmarks = [positions[len(positions) // 2]]
        else:
            landmarks = [
                positions[index * (len(positions) - 1) // (landmark_count - 1)]
                for index in range(landmark_count)
            ]
        for position in landmarks:
            candidates = [
                index
                for index, (start, end, new_start) in enumerate(values)
                if start + new_start <= position < end
            ]
            if not candidates:
                raise RuntimeError(
                    "no training window evaluates a priority position"
                )
            priority_indexes.append(
                min(
                    candidates,
                    key=lambda index: (
                        abs(
                            (
                                values[index][0]
                                + values[index][2]
                                + values[index][1]
                                - 1
                            )
                            / 2
                            - position
                        ),
                        index,
                    ),
                )
            )

    if maximum == 1 and not priority_indexes:
        return (values[(len(values) - 1) // 2],)

    selected = set(priority_indexes)
    # Preserve both boundaries whenever the cap permits.  This also makes the
    # no-priority path explicitly span the entire episode.
    for boundary in (0, len(values) - 1):
        if len(selected) < maximum:
            selected.add(boundary)
    if not selected:
        selected.add((len(values) - 1) // 2)

    # Greedy farthest-point completion is deterministic and avoids clustering
    # the non-event windows around either the event or one file boundary.
    while len(selected) < maximum:
        remaining = (index for index in range(len(values)) if index not in selected)
        chosen = max(
            remaining,
            key=lambda index: (min(abs(index - old) for old in selected), -index),
        )
        selected.add(chosen)
    return tuple(values[index] for index in sorted(selected))


def _load_cached_episode(
    cache: Path,
    entry: Mapping[str, Any],
    *,
    graph: ProcessGraph,
    targets: TargetContract,
    episodes: Mapping[str, Mapping[str, str]],
    profiles: Mapping[str, Mapping[str, str]],
    split_map: Mapping[str, str],
) -> _Episode:
    """Load one checksummed cache entry without enabling pickle deserialisation."""

    episode_id = str(entry["episode_id"])
    relative = Path(str(entry["file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe cache episode path: {relative}")
    with np.load(cache / relative, allow_pickle=False) as archive:
        required = {
            "x_raw",
            "controls_raw",
            "time_s",
            "dt_s",
            "class_target",
            "anomaly_target",
            "location_target",
            "mechanism_target",
            "behavior_active",
            "fault_active",
            "safety_event",
            "unsafe_forward_l",
        }
        if set(archive.files) != required:
            raise ValueError(f"cache array contract mismatch for {episode_id}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    rows = int(entry["row_count"])
    if rows < 1 or any(len(array) != rows for array in arrays.values()):
        raise ValueError(f"cache row count mismatch for {episode_id}")
    expected_shapes = {
        "x_raw": (rows, len(graph.features)),
        "controls_raw": (rows, len(CONTROL_FIELDS)),
        **{
            name: (rows,)
            for name in required - {"x_raw", "controls_raw"}
        },
    }
    wrong_shapes = [
        name for name, shape in expected_shapes.items() if arrays[name].shape != shape
    ]
    if wrong_shapes:
        raise ValueError(
            f"cache array shape mismatch for {episode_id}: {sorted(wrong_shapes)}"
        )
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"cache contains non-finite values for {episode_id}")
    target_limits = {
        "class_target": len(targets.codes),
        "anomaly_target": 2,
        "location_target": len(targets.locations),
        "mechanism_target": len(targets.mechanisms),
    }
    for name, limit in target_limits.items():
        array = arrays[name]
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(
            array >= limit
        ):
            raise ValueError(f"cache target range mismatch for {episode_id}: {name}")
    if episode_id not in episodes or episode_id not in split_map:
        raise ValueError(f"cache contains unknown episode {episode_id}")
    meta = episodes[episode_id]
    profile_id = str(meta["plant_profile_id"])
    if profile_id not in profiles:
        raise ValueError(f"cache episode {episode_id} references unknown profile")
    if str(entry["split"]) != split_map[episode_id]:
        raise ValueError(f"cache split mismatch for {episode_id}")
    return _Episode(
        episode_id=episode_id,
        split=split_map[episode_id],
        meta=meta,
        profile=profiles[profile_id],
        **arrays,
    )


class HTSTWindowDataset(IterableDataset[dict[str, Any]]):
    """Deterministic streaming windows with bounded counterfactual caching."""

    def __init__(
        self,
        dataset: Path,
        splits: Path,
        graph: ProcessGraph,
        targets: TargetContract,
        standardizer: FeatureStandardizer,
        *,
        accepted_splits: Sequence[str],
        window_size: int,
        stride: int,
        normal_only: bool = False,
        max_windows: int = 0,
        cache: Path | None = None,
        cache_manifest: Mapping[str, Any] | None = None,
        max_windows_per_episode: int = 0,
    ) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if stride < 1 or stride > window_size:
            raise ValueError("stride must be in [1, window_size]")
        if tuple(graph.features) != standardizer.features:
            raise ValueError("graph features and standardizer features differ")
        if max_windows < 0 or max_windows_per_episode < 0:
            raise ValueError("window limits must be non-negative")
        if cache_manifest is not None and cache is None:
            raise ValueError("cache_manifest requires a cache directory")
        self.dataset = Path(dataset)
        self.splits = Path(splits)
        self.graph = graph
        self.targets = targets
        self.standardizer = standardizer
        self.accepted_splits = frozenset(accepted_splits)
        self.window_size = window_size
        self.stride = stride
        self.normal_only = normal_only
        self.max_windows = max_windows
        self.cache = Path(cache) if cache is not None else None
        self.cache_manifest = cache_manifest
        self.max_windows_per_episode = max_windows_per_episode

    def _episode_pairs(
        self,
        episodes: Mapping[str, Mapping[str, str]],
        profiles: Mapping[str, Mapping[str, str]],
        split_map: Mapping[str, str],
    ) -> Iterator[tuple[_Episode, _Episode]]:
        accepted_episode_ids = frozenset(
            episode_id
            for episode_id, split in split_map.items()
            if split in self.accepted_splits
        )
        if self.cache is not None:
            # Local import avoids a module-level data <-> cache import cycle.
            from .cache import load_cache_manifest

            manifest = self.cache_manifest
            if manifest is None:
                manifest = load_cache_manifest(
                    self.cache,
                    self.dataset,
                    self.splits,
                    self.graph,
                    self.targets,
                )
            entries = list(manifest["episodes"])
            by_id = {str(item["episode_id"]): item for item in entries}
            if len(by_id) != len(entries) or set(by_id) != set(episodes):
                raise ValueError("cache manifest and episode metadata differ")
            current_group: str | None = None
            loaded: dict[str, _Episode] = {}
            for entry in entries:
                episode_id = str(entry["episode_id"])
                if episode_id not in accepted_episode_ids:
                    continue
                group = str(entry["counterfactual_group_id"])
                if group != current_group:
                    current_group = group
                    loaded = {}
                episode = _load_cached_episode(
                    self.cache,
                    entry,
                    graph=self.graph,
                    targets=self.targets,
                    episodes=episodes,
                    profiles=profiles,
                    split_map=split_map,
                )
                loaded[episode_id] = episode
                reference_id = str(entry["reference_episode_id"])
                if reference_id not in accepted_episode_ids:
                    raise ValueError(
                        f"split firewall rejected reference {reference_id!r} "
                        f"for accepted episode {episode_id!r}"
                    )
                reference = loaded.get(reference_id)
                if reference is None:
                    reference_entry = by_id.get(reference_id)
                    if reference_entry is None:
                        raise ValueError(
                            f"cache lacks reference {reference_id!r} for {episode_id}"
                        )
                    reference = _load_cached_episode(
                        self.cache,
                        reference_entry,
                        graph=self.graph,
                        targets=self.targets,
                        episodes=episodes,
                        profiles=profiles,
                        split_map=split_map,
                    )
                    loaded[reference_id] = reference
                yield episode, reference
            return

        current_group: str | None = None
        references: dict[str, _Episode] = {}
        for episode_id, signal_rows, label_rows in _joint_episode_rows(
            self.dataset,
            accepted_episode_ids=accepted_episode_ids,
        ):
            episode = _materialize_episode(
                episode_id,
                signal_rows,
                label_rows,
                graph=self.graph,
                targets=self.targets,
                episodes=episodes,
                profiles=profiles,
                split_map=split_map,
            )
            group = str(episode.meta["counterfactual_group_id"])
            if group != current_group:
                current_group = group
                references = {}
            scenario = str(episode.meta["canonical_scenario"])
            reference_name = str(episode.meta.get("counterfactual_reference", "normal"))
            if scenario == reference_name:
                reference = episode
            else:
                reference = references.get(reference_name)
                if reference is None:
                    raise ValueError(
                        f"counterfactual reference {reference_name!r} precedes no episode "
                        f"in group {group}"
                    )
            references[scenario] = episode
            yield episode, reference

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers != 1:
            raise RuntimeError(
                "HTSTWindowDataset uses a single deterministic CSV stream; set num_workers=0"
            )
        episodes, profiles, split_map = _read_metadata(self.dataset, self.splits)
        emitted = 0
        for episode, reference in self._episode_pairs(episodes, profiles, split_map):
            episode_id = episode.episode_id
            if episode.split not in self.accepted_splits:
                continue
            if self.normal_only and str(episode.meta["canonical_code"]) not in NORMAL_BEHAVIOR_CODES:
                continue
            if reference.x_raw.shape != episode.x_raw.shape or not np.array_equal(
                reference.time_s, episode.time_s
            ):
                raise ValueError(f"misaligned counterfactual pair for {episode_id}")
            x = self.standardizer.transform(episode.x_raw).astype(np.float32)
            reference_x = self.standardizer.transform(reference.x_raw).astype(np.float32)
            edge_volumes = np.asarray(
                self.graph.edge_volumes_for_profile(episode.profile), dtype=np.float32
            )
            ranges = _sample_window_ranges(
                tuple(_window_ranges(len(x), self.window_size, self.stride)),
                self.max_windows_per_episode,
                priority_positions=np.flatnonzero(
                    episode.class_target != self.targets.normal_index
                ),
            )
            for start, end, new_start in ranges:
                yield self._build_window(
                    episode,
                    reference,
                    x,
                    reference_x,
                    edge_volumes,
                    start,
                    end,
                    new_start,
                )
                emitted += 1
                if self.max_windows and emitted >= self.max_windows:
                    return

    def _build_window(
        self,
        episode: _Episode,
        reference: _Episode,
        x: np.ndarray,
        reference_x: np.ndarray,
        edge_volumes: np.ndarray,
        start: int,
        end: int,
        new_start: int,
    ) -> dict[str, Any]:
        width = end - start
        w = self.window_size

        def padded(source: np.ndarray, fill: float = 0.0) -> np.ndarray:
            shape = (w, *source.shape[1:])
            result = np.full(shape, fill, dtype=source.dtype)
            result[:width] = source[start:end]
            return result

        valid_mask = np.zeros(w, dtype=np.bool_)
        valid_mask[:width] = True
        eval_mask = np.zeros(w, dtype=np.bool_)
        eval_mask[new_start:width] = True
        class_target = np.full(w, -100, dtype=np.int64)
        anomaly_target = np.full(w, -100, dtype=np.int64)
        location_target = np.full(w, -100, dtype=np.int64)
        mechanism_target = np.full(w, -100, dtype=np.int64)
        for result, source in (
            (class_target, episode.class_target),
            (anomaly_target, episode.anomaly_target),
            (location_target, episode.location_target),
            (mechanism_target, episode.mechanism_target),
        ):
            result[:width] = source[start:end]
        time = np.full(w, np.nan, dtype=np.float32)
        time[:width] = episode.time_s[start:end]
        effect_time = _float_or_nan(episode.meta.get("behavior_effect_time_s", ""))
        return {
            "x": torch.from_numpy(padded(x)),
            "reference_x": torch.from_numpy(padded(reference_x)),
            "controls_raw": torch.from_numpy(padded(episode.controls_raw)),
            "reference_controls_raw": torch.from_numpy(
                padded(reference.controls_raw)
            ),
            "dt_s": torch.from_numpy(padded(episode.dt_s, fill=1.0)),
            "edge_volume_l": torch.from_numpy(edge_volumes.copy()),
            "valid_mask": torch.from_numpy(valid_mask),
            "eval_mask": torch.from_numpy(eval_mask),
            "class_target": torch.from_numpy(class_target),
            "anomaly_target": torch.from_numpy(anomaly_target),
            "location_target": torch.from_numpy(location_target),
            "mechanism_target": torch.from_numpy(mechanism_target),
            "behavior_active": torch.from_numpy(padded(episode.behavior_active)),
            "fault_active": torch.from_numpy(padded(episode.fault_active)),
            "safety_event": torch.from_numpy(padded(episode.safety_event)),
            "unsafe_forward_l": torch.from_numpy(padded(episode.unsafe_forward_l)),
            "time_s": torch.from_numpy(time),
            "episode_id": episode.episode_id,
            "plant_profile_id": str(episode.meta["plant_profile_id"]),
            "counterfactual_group_id": str(episode.meta["counterfactual_group_id"]),
            "canonical_code": str(episode.meta["canonical_code"]),
            "split": episode.split,
            "effect_time_s": effect_time,
            "detection_eligible": int(episode.meta.get("detection_eligible", "0")),
        }


def normalized_sensor_uncertainty(
    graph: ProcessGraph, standardizer: FeatureStandardizer
) -> np.ndarray:
    raw = np.asarray(graph.sensor_standard_uncertainty_raw, dtype=np.float64)
    known = np.asarray(graph.sensor_uncertainty_known, dtype=bool)
    result = np.zeros_like(raw)
    result[known] = raw[known] / standardizer.scale[known]
    return result.astype(np.float32)


__all__ = [
    "DATA_LOADER_VERSION",
    "FeatureStandardizer",
    "HTSTWindowDataset",
    "JOIN_KEYS",
    "NORMAL_BEHAVIOR_CODES",
    "TargetContract",
    "_sample_window_ranges",
    "load_target_contract",
    "normalized_sensor_uncertainty",
]
