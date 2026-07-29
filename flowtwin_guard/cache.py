"""One-pass CSV-to-episode cache for scalable FlowTwin experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline_common import (
    BASE_DIR,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
)

from .data import (
    FeatureStandardizer,
    TargetContract,
    _joint_episode_rows,
    _materialize_episode,
    _read_metadata,
)
from .graph import ProcessGraph


CACHE_VERSION = "0.3.0"
CACHE_ARRAYS = (
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
)

MATERIALIZATION_SOURCES = (
    "build_flowtwin_cache.py",
    "flowtwin_guard/cache.py",
    "flowtwin_guard/data.py",
    "flowtwin_guard/graph.py",
    "ml_pipeline_common.py",
    "ml_contract.json",
    "fault_asset_contract.json",
    "reference_pid.json",
)


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = len(values)
    if batch_count == 0:
        return count, mean, m2
    batch_mean = values.astype(np.float64).mean(axis=0)
    centered = values.astype(np.float64) - batch_mean
    batch_m2 = np.square(centered).sum(axis=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + np.square(delta) * count * batch_count / total
    return total, merged_mean, merged_m2


def _episode_filename(index: int, episode_id: str) -> str:
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:16]
    return f"episodes/E{index:05d}-{digest}.npz"


def build_episode_cache(
    dataset: Path,
    splits: Path,
    output: Path,
    graph: ProcessGraph,
    targets: TargetContract,
) -> Path:
    """Convert joined CSV episodes once and publish a checksummed cache atomically."""

    dataset = Path(dataset)
    splits = Path(splits)
    output = Path(output)
    for directory in (dataset, splits):
        failures = verify_checksums(directory)
        if failures:
            raise ValueError(f"checksum verification failed for {directory}: {failures}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"cache output is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".flowtwin-cache-", dir=output.parent))
    (temporary / "episodes").mkdir()
    try:
        episode_meta, profiles, split_map = _read_metadata(dataset, splits)
        entries: list[dict[str, Any]] = []
        relative_files: list[str] = []
        scenario_ids: dict[str, str] = {}
        current_group: str | None = None
        count = 0
        mean = np.zeros(len(graph.features), dtype=np.float64)
        m2 = np.zeros(len(graph.features), dtype=np.float64)
        row_count = 0
        for index, (episode_id, signals, labels) in enumerate(
            _joint_episode_rows(dataset)
        ):
            episode = _materialize_episode(
                episode_id,
                signals,
                labels,
                graph=graph,
                targets=targets,
                episodes=episode_meta,
                profiles=profiles,
                split_map=split_map,
            )
            group = str(episode.meta["counterfactual_group_id"])
            if group != current_group:
                current_group = group
                scenario_ids = {}
            scenario = str(episode.meta["canonical_scenario"])
            reference_name = str(
                episode.meta.get("counterfactual_reference", "normal")
            )
            if scenario == reference_name:
                reference_episode_id = episode_id
            else:
                reference_episode_id = scenario_ids.get(reference_name, "")
                if not reference_episode_id:
                    raise ValueError(
                        f"missing {reference_name!r} reference before {episode_id}"
                    )
            scenario_ids[scenario] = episode_id
            relative = _episode_filename(index, episode_id)
            np.savez(
                temporary / relative,
                **{name: getattr(episode, name) for name in CACHE_ARRAYS},
            )
            relative_files.append(relative)
            rows = len(episode.x_raw)
            row_count += rows
            if episode.split == "train_id":
                count, mean, m2 = _merge_moments(
                    count, mean, m2, episode.x_raw
                )
            entries.append(
                {
                    "episode_id": episode_id,
                    "file": relative,
                    "row_count": rows,
                    "split": episode.split,
                    "plant_profile_id": str(episode.meta["plant_profile_id"]),
                    "counterfactual_group_id": group,
                    "canonical_code": str(episode.meta["canonical_code"]),
                    "canonical_scenario": scenario,
                    "counterfactual_reference": reference_name,
                    "reference_episode_id": reference_episode_id,
                }
            )
        if not entries or count < 2:
            raise ValueError("cache source has no episodes or insufficient train rows")
        scale = np.sqrt(np.maximum(m2 / (count - 1), 0.0))
        scale = np.maximum(scale, 1e-6)
        standardizer = FeatureStandardizer(
            tuple(graph.features), count, mean, scale
        )
        manifest = {
            "cache_version": CACHE_VERSION,
            "dataset_manifest_sha256": sha256_file(
                dataset / "dataset_manifest.json"
            ),
            "split_manifest_sha256": sha256_file(splits / "split_manifest.json"),
            "dataset_checksums_sha256": sha256_file(dataset / "checksums.sha256"),
            "split_checksums_sha256": sha256_file(splits / "checksums.sha256"),
            "graph_version": graph.version,
            "plant_id": graph.plant_id,
            "features": list(graph.features),
            "target_codes": list(targets.codes),
            "array_contract": list(CACHE_ARRAYS),
            "counts": {
                "episodes": len(entries),
                "rows": row_count,
                "train_rows_for_scaling": count,
            },
            "standardizer": standardizer.to_dict(),
            "provenance": {
                name: sha256_file(BASE_DIR / name)
                for name in MATERIALIZATION_SOURCES
            },
            "episodes": entries,
            "claim_boundary": (
                "Performance cache only; oracle arrays remain targets/evaluation fields "
                "and must never be concatenated to x_raw or controls_raw."
            ),
            "artifacts": ["cache_manifest.json", *relative_files, "checksums.sha256"],
        }
        write_json(temporary / "cache_manifest.json", manifest)
        write_checksums(temporary, ["cache_manifest.json", *relative_files])
        failures = verify_checksums(temporary)
        if failures:
            raise ValueError(f"new cache failed checksum verification: {failures}")
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_cache_manifest(
    cache: Path,
    dataset: Path,
    splits: Path,
    graph: ProcessGraph,
    targets: TargetContract,
) -> dict[str, Any]:
    cache = Path(cache)
    failures = verify_checksums(cache)
    if failures:
        raise ValueError(f"cache checksum verification failed: {failures}")
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "cache_version": (manifest.get("cache_version"), CACHE_VERSION),
        "dataset_manifest_sha256": (
            manifest.get("dataset_manifest_sha256"),
            sha256_file(Path(dataset) / "dataset_manifest.json"),
        ),
        "split_manifest_sha256": (
            manifest.get("split_manifest_sha256"),
            sha256_file(Path(splits) / "split_manifest.json"),
        ),
        "dataset_checksums_sha256": (
            manifest.get("dataset_checksums_sha256"),
            sha256_file(Path(dataset) / "checksums.sha256"),
        ),
        "split_checksums_sha256": (
            manifest.get("split_checksums_sha256"),
            sha256_file(Path(splits) / "checksums.sha256"),
        ),
        "graph_version": (manifest.get("graph_version"), graph.version),
        "features": (tuple(manifest.get("features", [])), graph.features),
        "target_codes": (tuple(manifest.get("target_codes", [])), targets.codes),
        "array_contract": (tuple(manifest.get("array_contract", [])), CACHE_ARRAYS),
    }
    bad = [name for name, (actual, expected) in checks.items() if actual != expected]
    if bad:
        raise ValueError(f"cache contract mismatch: {bad}")
    expected_provenance = {
        name: sha256_file(BASE_DIR / name) for name in MATERIALIZATION_SOURCES
    }
    if manifest.get("provenance") != expected_provenance:
        raise ValueError(
            "cache materialization provenance no longer matches current source"
        )
    return manifest


__all__ = [
    "CACHE_ARRAYS",
    "CACHE_VERSION",
    "MATERIALIZATION_SOURCES",
    "build_episode_cache",
    "load_cache_manifest",
]
