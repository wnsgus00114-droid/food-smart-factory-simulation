#!/usr/bin/env python3
"""Run the frozen FlowTwin-Guard neural benchmark and ablation matrix.

The runner enforces a train/validation/test firewall: fixed-budget fitting uses
``train_id`` only, every threshold and calibration object is finalized on
``validation_id``, and only then is any ``test_*`` iterator constructed.
Subset or budget-overridden runs remain useful, but are labelled development
runs rather than confirmatory evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "FlowTwin benchmark dependencies are missing. Create the documented "
        "Python 3.12 environment and install requirements-ml.txt."
    ) from exc

from flowtwin_guard.ablation import (
    ABLATION_NAMES,
    ABLATION_VERSION,
    build_ablation_model,
)
from flowtwin_guard.alarm import (
    ALARM_POLICY_VERSION,
    AlarmPolicyConfig,
    apply_alarm_policy,
    fit_alarm_policy,
)
from flowtwin_guard.baselines import (
    AVAILABLE_BASELINES,
    BASELINE_VERSION,
    BaselineConfig,
    build_baseline,
)
from flowtwin_guard.cache import load_cache_manifest
from flowtwin_guard.conformal import (
    ModeConformalCalibrator,
    counterfactual_group_mode_block_max,
)
from flowtwin_guard.data import (
    FeatureStandardizer,
    HTSTWindowDataset,
    TargetContract,
    load_target_contract,
    normalized_sensor_uncertainty,
)
from flowtwin_guard.dspr import (
    DSPR_DIAGNOSTIC_ADAPTATION_NAME,
    DSPRDiagnosticConfig,
    build_dspr_diagnostic_adaptation,
)
from flowtwin_guard.graph import CONTROL_FIELDS, ProcessGraph, build_process_graph
from flowtwin_guard.hybrid import (
    HYBRID_MODEL_VERSION,
    FlowTwinHybrid,
    FlowTwinHybridConfig,
)
from flowtwin_guard.metrics import (
    METRICS_VERSION,
    classification_metrics,
    conformal_metrics,
    event_detection_metrics,
    false_alarm_metrics,
    ood_detection_metrics,
    risk_coverage_curve,
    volume_metrics,
)
from flowtwin_guard.model import (
    MODEL_VERSION,
    FlowTwinConfig,
    RouteGate,
    counterfactual_contrastive_loss,
    focal_cross_entropy,
)
from ml_pipeline_common import (
    BASE_DIR,
    read_csv_rows,
    sha256_file,
    sha256_json,
    verify_checksums,
    write_checksums,
    write_json,
)


BENCHMARK_RUNNER_VERSION = "0.3.0"
BENCHMARK_CONTRACT_VERSION = "0.2.0"
DEFAULT_CONTRACT = BASE_DIR / "flowtwin_benchmark_contract.json"
V03_CANDIDATE_CONTRACT_VERSION = "0.1.0"
DEFAULT_V03_CANDIDATE_CONTRACT = BASE_DIR / "flowtwin_v03_candidate_contract.json"
FULL_MODEL_NAME = "flowtwin_guard"
V03_CANDIDATE_NAME = "flowtwin_hybrid_v03_dev"
ABLATION_PREFIX = "ablation:"


@dataclass(frozen=True)
class Variant:
    """One architecture/protocol row in the registered comparison matrix."""

    variant_id: str
    family: str
    model_name: str
    ablation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "family": self.family,
            "model_name": self.model_name,
            "ablation": self.ablation,
        }


@dataclass
class RuntimeContext:
    dataset: Path
    splits: Path
    cache: Path | None
    cache_manifest: Mapping[str, Any] | None
    graph: ProcessGraph
    targets: TargetContract
    standardizer: FeatureStandardizer
    budget: Mapping[str, Any]
    device: torch.device
    candidate_contract: Mapping[str, Any] | None = None


def load_benchmark_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load and strictly validate the benchmark registration document."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("benchmark_contract_version") != BENCHMARK_CONTRACT_VERSION:
        raise ValueError("unsupported flowtwin benchmark contract version")
    required = {
        "benchmark_id",
        "execution_tiers",
        "dataset_contract",
        "registered_models",
        "registered_ablations",
        "training_budget",
        "reference_execution",
        "loss_weights",
        "calibration",
        "endpoints",
        "aggregation",
        "claim_boundary",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"benchmark contract is missing {missing}")
    models = tuple(map(str, value["registered_models"]))
    expected_models = (
        FULL_MODEL_NAME,
        *AVAILABLE_BASELINES,
        DSPR_DIAGNOSTIC_ADAPTATION_NAME,
    )
    if models != expected_models:
        raise ValueError(
            f"registered_models must be exactly {expected_models}, got {models}"
        )
    ablations = tuple(map(str, value["registered_ablations"]))
    if ablations != tuple(ABLATION_NAMES):
        raise ValueError("registered_ablations do not match the implemented registry")
    seeds = value["training_budget"].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("training_budget.seeds must be unique integer values")
    if value["dataset_contract"].get("calibration_split") != ["validation_id"]:
        raise ValueError("calibration_split must be exactly validation_id")
    if value["dataset_contract"].get("training_split") != ["train_id"]:
        raise ValueError("training_split must be exactly train_id")
    if value["reference_execution"].get("device") != "cpu":
        raise ValueError("the registered reference device must remain cpu")
    return value


def load_v03_candidate_contract(
    path: Path = DEFAULT_V03_CANDIDATE_CONTRACT,
) -> dict[str, Any]:
    """Load the explicitly post-hoc v0.3 development contract.

    This contract never extends the frozen v0.2 registry.  It binds the
    already-opened D2 development artifacts, the matched comparison budget,
    the train-only transport-context gate, and the validation-only alarm grid.
    """

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "candidate_contract_version",
        "candidate_id",
        "status",
        "d2_test_seen_before_freeze",
        "base_benchmark_contract",
        "development_dataset",
        "model_contract",
        "training_budget",
        "critical_delay_gate",
        "calibration",
        "alarm_policy",
        "planned_comparison_variants",
        "future_confirmatory_gate",
        "claim_boundary",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"v0.3 candidate contract is missing {missing}")
    if value["candidate_contract_version"] != V03_CANDIDATE_CONTRACT_VERSION:
        raise ValueError("unsupported v0.3 candidate contract version")
    if value["candidate_id"] != V03_CANDIDATE_NAME:
        raise ValueError("v0.3 candidate_id does not match the implemented model")
    if value["status"] != "development_only" or not bool(
        value["d2_test_seen_before_freeze"]
    ):
        raise ValueError(
            "v0.3 contract must disclose development_only and opened-D2 status"
        )
    parent = value["base_benchmark_contract"]
    if parent.get("version") != BENCHMARK_CONTRACT_VERSION:
        raise ValueError("v0.3 parent benchmark version mismatch")
    dataset = value["development_dataset"]
    if dataset.get("test_status") != (
        "opened development set; never eligible for confirmatory evidence"
    ):
        raise ValueError("v0.3 D2 test-status disclosure is missing")
    if dataset.get("feature_set") != value["training_budget"].get("feature_set"):
        raise ValueError("v0.3 dataset and budget feature sets differ")
    _validate_budget(value["training_budget"])
    gate = value["critical_delay_gate"]
    expected_edges = ("L-005", "L-007", "L-008", "L-009", "L-010")
    if gate.get("fit_split") != "train_id" or tuple(gate.get("edge_tags", ())) != expected_edges:
        raise ValueError("v0.3 critical-delay gate must be train-only exact FIFO")
    for name in (
        "minimum_exhaustive_per_edge_coverage",
        "minimum_sampled_train_per_edge_coverage",
    ):
        threshold = float(gate[name])
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"v0.3 {name} must be in (0,1]")
    alarm = value["alarm_policy"]
    if alarm.get("fit_split") != "validation_id":
        raise ValueError("v0.3 operational alarm must fit validation_id only")
    if alarm.get("apply_to") != "every_selected_variant_in_candidate_run":
        raise ValueError("v0.3 alarm policy must apply to every selected variant")
    if alarm.get("no_operating_point_policy") != (
        "fail before any test iterator is constructed"
    ):
        raise ValueError("v0.3 alarm policy must fail closed without fallback")
    planned = tuple(map(str, value["planned_comparison_variants"]))
    if not planned or planned[0] != V03_CANDIDATE_NAME:
        raise ValueError("v0.3 planned comparison must begin with the candidate")
    return value


def registered_variants(contract: Mapping[str, Any]) -> tuple[Variant, ...]:
    """Return the frozen model rows in their publication-table order."""

    result = [Variant(FULL_MODEL_NAME, "proposed", FULL_MODEL_NAME)]
    result.extend(
        Variant(name, "neural_baseline", name)
        for name in map(str, contract["registered_models"])
        if name != FULL_MODEL_NAME
    )
    result.extend(
        Variant(
            f"{ABLATION_PREFIX}{name}",
            "flowtwin_ablation",
            FULL_MODEL_NAME,
            name,
        )
        for name in map(str, contract["registered_ablations"])
    )
    return tuple(result)


def _validate_split_domain_contract(dataset: Path, splits: Path) -> dict[str, Any]:
    """Cross-check episode domain truth against its assigned split, fail closed."""

    episode_rows = list(read_csv_rows(dataset / "episodes.csv"))
    profile_rows = list(read_csv_rows(dataset / "profiles.csv"))
    split_rows = list(read_csv_rows(splits / "episode_splits.csv"))
    episodes = {str(row["episode_id"]): row for row in episode_rows}
    profiles = {str(row["plant_profile_id"]): row for row in profile_rows}
    assignments = {str(row["episode_id"]): row for row in split_rows}
    if len(episodes) != len(episode_rows) or len(assignments) != len(split_rows):
        raise ValueError("episode or split table contains duplicate episode_id values")
    if len(profiles) != len(profile_rows):
        raise ValueError("profile table contains duplicate plant_profile_id values")
    if set(episodes) != set(assignments):
        raise ValueError("episode and split tables do not contain identical episode IDs")
    allowed_splits = {"train_id", "validation_id", "test_id", "test_ood_profile"}
    profile_splits: dict[str, set[str]] = {}
    profile_domains: dict[str, set[str]] = {}
    profile_hashes: dict[str, set[str]] = {}
    group_splits: dict[str, set[str]] = {}
    domains: Counter[str] = Counter()
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        assignment = assignments[episode_id]
        for field in (
            "plant_profile_id",
            "counterfactual_group_id",
            "profile_config_hash",
            "noise_seed",
            "sampling_seed",
            "canonical_code",
            "domain",
        ):
            if str(episode.get(field, "")) != str(assignment.get(field, "")):
                raise ValueError(
                    f"episode/split metadata mismatch for {episode_id}: {field}"
                )
        domain = str(episode["domain"])
        split = str(assignment["split"])
        if split not in allowed_splits:
            raise ValueError(f"unsupported split name {split!r}")
        if domain == "ID" and split == "test_ood_profile":
            raise ValueError("ID episode cannot be assigned to test_ood_profile")
        if domain != "ID" and split != "test_ood_profile":
            raise ValueError("non-ID episode must be assigned to test_ood_profile")
        profile = str(episode["plant_profile_id"])
        group = str(episode["counterfactual_group_id"])
        profile_splits.setdefault(profile, set()).add(split)
        profile_domains.setdefault(profile, set()).add(domain)
        profile_hashes.setdefault(profile, set()).add(
            str(episode["profile_config_hash"])
        )
        group_splits.setdefault(group, set()).add(split)
        domains[domain] += 1
    if set(profiles) != set(profile_splits):
        raise ValueError("profile and episode tables do not contain identical profile IDs")
    if any(len(values) != 1 for values in profile_splits.values()):
        raise ValueError("plant_profile_id crosses split partitions")
    if any(len(values) != 1 for values in profile_domains.values()):
        raise ValueError("plant_profile_id crosses domain partitions")
    if any(len(values) != 1 for values in profile_hashes.values()):
        raise ValueError("plant_profile_id has inconsistent profile_config_hash values")
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("counterfactual_group_id crosses split partitions")

    profile_metadata = {
        "plant_profile_id",
        "profile_index",
        "profile_seed",
        "profile_config_hash",
        "domain",
    }
    profile_fields = sorted(set(profile_rows[0]) - profile_metadata) if profile_rows else []
    if not profile_fields:
        raise ValueError("profiles.csv contains no physical profile parameters")
    profile_values: dict[str, dict[str, float]] = {}
    has_profile_domain = "domain" in profile_rows[0]
    for profile_id, row in profiles.items():
        try:
            values = {name: float(row[name]) for name in profile_fields}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"profile {profile_id} contains a missing or non-numeric parameter"
            ) from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"profile {profile_id} contains a non-finite parameter")
        expected_hash = sha256_json(values)
        table_hash = str(row.get("profile_config_hash", ""))
        episode_hash = next(iter(profile_hashes[profile_id]))
        if table_hash != expected_hash or episode_hash != expected_hash:
            raise ValueError(
                f"profile_config_hash does not match physical parameters for {profile_id}"
            )
        episode_domain = next(iter(profile_domains[profile_id]))
        if has_profile_domain and str(row.get("domain", "")) != episode_domain:
            raise ValueError(f"profile/episode domain mismatch for {profile_id}")
        profile_values[profile_id] = values

    observed_ood = sorted(name for name in domains if name != "ID")
    domain_contract_verified = not observed_ood
    profile_table_domain_verified = not observed_ood or has_profile_domain
    profile_parameter_ranges_verified = not observed_ood
    domain_contract: Mapping[str, Any] | None = None
    if observed_ood:
        if not has_profile_domain:
            raise ValueError("non-ID data require an explicit profiles.csv domain column")
        manifest = json.loads(
            (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        candidate = manifest.get("domain_contract")
        if not isinstance(candidate, Mapping):
            raise ValueError(
                "non-ID data require dataset_manifest.domain_contract provenance"
            )
        required = {
            "version",
            "id_domain",
            "ood_domains",
            "generation_policy",
            "profile_parameter_ranges",
            "source_contract",
            "synthetic_only",
        }
        if required - candidate.keys():
            raise ValueError("dataset domain_contract is incomplete")
        if candidate["id_domain"] != "ID" or sorted(
            map(str, candidate["ood_domains"])
        ) != observed_ood:
            raise ValueError("dataset domain_contract does not match observed domains")
        if candidate["synthetic_only"] is not True:
            raise ValueError("OOD domain_contract must declare synthetic_only=true")
        if not isinstance(candidate["generation_policy"], Mapping):
            raise ValueError("OOD domain_contract generation_policy must be an object")
        source = candidate["source_contract"]
        if not isinstance(source, Mapping):
            raise ValueError("OOD domain_contract source_contract must be an object")
        source_name = str(source.get("filename", ""))
        source_hash = str(source.get("sha256", ""))
        if (
            not source_name
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError("OOD domain_contract source provenance is invalid")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping) or str(
            provenance.get(source_name, "")
        ) != source_hash:
            raise ValueError("OOD source-contract hash is not bound to dataset provenance")

        range_boxes = candidate["profile_parameter_ranges"]
        if not isinstance(range_boxes, Mapping) or not range_boxes:
            raise ValueError("OOD domain_contract requires parameter range provenance")
        expected_domains = {"ID", *observed_ood}
        if set(map(str, range_boxes)) != expected_domains:
            raise ValueError("OOD parameter-range domains do not match observed domains")
        normalized_ranges: dict[str, dict[str, tuple[float, float]]] = {}
        for domain_name in sorted(expected_domains):
            box = range_boxes[domain_name]
            if not isinstance(box, Mapping) or set(map(str, box)) != set(profile_fields):
                raise ValueError(
                    f"{domain_name} parameter ranges do not match profiles.csv fields"
                )
            normalized_ranges[domain_name] = {}
            for field in profile_fields:
                bounds = box[field]
                if (
                    not isinstance(bounds, (list, tuple))
                    or len(bounds) != 2
                    or isinstance(bounds[0], bool)
                    or isinstance(bounds[1], bool)
                ):
                    raise ValueError(f"invalid range for {domain_name}.{field}")
                try:
                    lower, upper = float(bounds[0]), float(bounds[1])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid range for {domain_name}.{field}"
                    ) from exc
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError(f"invalid range for {domain_name}.{field}")
                normalized_ranges[domain_name][field] = (lower, upper)

        domain_names = sorted(expected_domains)
        for left_index, left_name in enumerate(domain_names):
            for right_name in domain_names[left_index + 1 :]:
                separated = any(
                    normalized_ranges[left_name][field][1]
                    < normalized_ranges[right_name][field][0]
                    or normalized_ranges[right_name][field][1]
                    < normalized_ranges[left_name][field][0]
                    for field in profile_fields
                )
                if not separated:
                    raise ValueError(
                        f"profile supports {left_name} and {right_name} are not strictly disjoint"
                    )

        for profile_id, values in profile_values.items():
            profile_domain = next(iter(profile_domains[profile_id]))
            box = normalized_ranges[profile_domain]
            outside = [
                field
                for field, value in values.items()
                if value < box[field][0] or value > box[field][1]
            ]
            if outside:
                raise ValueError(
                    f"profile {profile_id} is outside declared {profile_domain} ranges: {outside}"
                )
        domain_contract = candidate
        domain_contract_verified = True
        profile_table_domain_verified = True
        profile_parameter_ranges_verified = True
    return {
        "status": "pass",
        "episodes": len(episodes),
        "profiles": len(profile_splits),
        "counterfactual_groups": len(group_splits),
        "domain_episode_counts": dict(sorted(domains.items())),
        "observed_ood_domains": observed_ood,
        "domain_contract_verified": domain_contract_verified,
        "profile_table_domain_verified": profile_table_domain_verified,
        "profile_parameter_ranges_verified": profile_parameter_ranges_verified,
        "domain_contract": domain_contract,
        "mapping_policy": "ID -> *_id; non-ID -> test_ood_profile",
    }


def resolve_variants(
    requested: Sequence[str] | None, contract: Mapping[str, Any]
) -> tuple[Variant, ...]:
    """Resolve frozen aliases plus the explicit development-only candidate.

    ``all``, ``models`` and ``ablations`` deliberately retain their frozen
    v0.2 meaning.  The v0.3 row is reachable only by its full identifier and
    is validated against a separate development contract by :func:`run`.
    """

    registered = registered_variants(contract)
    candidate = Variant(
        V03_CANDIDATE_NAME,
        "experimental_candidate",
        V03_CANDIDATE_NAME,
    )
    catalog = (*registered, candidate)
    by_id = {item.variant_id: item for item in catalog}
    tokens = list(requested or ("all",))
    selected: list[str] = []
    for token in tokens:
        if token == "all":
            selected.extend(item.variant_id for item in registered)
        elif token == "models":
            selected.extend(
                item.variant_id
                for item in registered
                if item.family in {"proposed", "neural_baseline"}
            )
        elif token == "ablations":
            selected.append(FULL_MODEL_NAME)
            selected.extend(
                item.variant_id
                for item in registered
                if item.family == "flowtwin_ablation"
            )
        elif token in by_id:
            selected.append(token)
        elif token in ABLATION_NAMES:
            selected.append(f"{ABLATION_PREFIX}{token}")
        else:
            raise ValueError(
                f"unknown --variants value {token!r}; use all, models, ablations, "
                f"or one of {sorted(by_id)}"
            )
    selected_set = set(selected)
    if not selected_set:
        raise ValueError("at least one benchmark variant is required")
    return tuple(item for item in catalog if item.variant_id in selected_set)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        help=(
            "explicit post-hoc v0.3 development contract; required implicitly "
            "when flowtwin_hybrid_v03_dev is selected"
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help=(
            "registered IDs or aliases: all, models (full + six baselines), "
            "ablations (full + nine ablations); the v0.3 candidate requires "
            "its full explicit identifier"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--feature-set")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--train-windows-per-episode", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--observer-epochs", type=int)
    parser.add_argument("--epochs", type=int, dest="diagnostic_epochs")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--observer-hidden-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--ood-alpha", type=float)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="save per-row compressed arrays for paired downstream analyses",
    )
    return parser


def _effective_budget(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_contract is None:
        budget = dict(contract["training_budget"])
        budget["feature_set"] = contract["dataset_contract"]["feature_set"]
    else:
        budget = dict(candidate_contract["training_budget"])
    mapping = {
        "feature_set": "feature_set",
        "window_size": "window_size",
        "stride": "stride",
        "train_windows_per_episode": "train_windows_per_episode",
        "batch_size": "batch_size",
        "observer_epochs": "observer_epochs",
        "diagnostic_epochs": "diagnostic_epochs",
        "hidden_dim": "hidden_dim",
        "observer_hidden_dim": "observer_hidden_dim",
        "layers": "layers",
        "attention_heads": "attention_heads",
        "dropout": "dropout",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
    }
    for argument, key in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            budget[key] = value
    calibration = (
        contract["calibration"]
        if candidate_contract is None
        else candidate_contract["calibration"]
    )
    default_alpha = budget.get("alpha", calibration["conformal_alpha"])
    default_ood_alpha = budget.get("ood_alpha", calibration["ood_alpha"])
    budget["alpha"] = float(args.alpha) if args.alpha is not None else float(default_alpha)
    budget["ood_alpha"] = (
        float(args.ood_alpha) if args.ood_alpha is not None else float(default_ood_alpha)
    )
    budget["seeds"] = list(args.seeds or budget["seeds"])
    return budget


def _validate_budget(budget: Mapping[str, Any]) -> None:
    positive_integer = (
        "window_size",
        "stride",
        "train_windows_per_episode",
        "batch_size",
        "observer_epochs",
        "diagnostic_epochs",
        "hidden_dim",
        "observer_hidden_dim",
        "layers",
        "attention_heads",
    )
    for name in positive_integer:
        value = budget[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if budget["stride"] > budget["window_size"]:
        raise ValueError("stride cannot exceed window_size")
    if budget["hidden_dim"] % budget["attention_heads"]:
        raise ValueError("hidden_dim must be divisible by attention_heads")
    if not 0.0 <= float(budget["dropout"]) < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if float(budget["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(budget["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    for name in ("alpha", "ood_alpha"):
        if not 0.0 < float(budget[name]) < 1.0:
            raise ValueError(f"{name} must be in (0,1)")
    seeds = list(budget["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty unique list")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(name)


def _prepare_output(path: Path) -> tuple[Path, bool]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existed_empty = False
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        existed_empty = True
    temporary = Path(tempfile.mkdtemp(prefix=".flowtwin-benchmark-", dir=output.parent))
    return temporary, existed_empty


def _commit_output(temporary: Path, output: Path, existed_empty: bool) -> None:
    output = output.resolve()
    if existed_empty:
        output.rmdir()
    os.replace(temporary, output)


def _dataset(
    context: RuntimeContext,
    accepted_splits: Sequence[str],
    *,
    training: bool = False,
    normal_only: bool = False,
) -> HTSTWindowDataset:
    """Build a deterministic iterator; evaluation is always exhaustive."""

    return HTSTWindowDataset(
        context.dataset,
        context.splits,
        context.graph,
        context.targets,
        context.standardizer,
        accepted_splits=accepted_splits,
        window_size=int(context.budget["window_size"]),
        stride=int(context.budget["stride"]),
        normal_only=normal_only,
        max_windows=0,
        cache=context.cache,
        cache_manifest=context.cache_manifest,
        max_windows_per_episode=(
            int(context.budget["train_windows_per_episode"]) if training else 0
        ),
    )


def _loader(dataset: HTSTWindowDataset, context: RuntimeContext) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(context.budget["batch_size"]),
        num_workers=0,
        pin_memory=False,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _target_weights(
    context: RuntimeContext,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Derive shared weights from every train row, counted exactly once."""

    widths = {
        "class": len(context.targets.codes),
        "anomaly": 2,
        "location": len(context.targets.locations),
        "mechanism": len(context.targets.mechanisms),
    }
    fields = {
        "class": "class_target",
        "anomaly": "anomaly_target",
        "location": "location_target",
        "mechanism": "mechanism_target",
    }
    counts = {name: np.zeros(width, dtype=np.int64) for name, width in widths.items()}
    for batch in _loader(_dataset(context, ("train_id",), training=False), context):
        selected = batch["valid_mask"] & batch["eval_mask"]
        for name, field in fields.items():
            values = batch[field][selected].numpy()
            counts[name] += np.bincount(values, minlength=widths[name])
    if counts["class"].sum() == 0:
        raise ValueError("training split produced no rows for class weighting")
    weights: dict[str, Tensor] = {}
    report: dict[str, Any] = {}
    for name, values in counts.items():
        observed = values > 0
        raw = np.zeros(len(values), dtype=np.float64)
        raw[observed] = np.sqrt(
            values[observed].sum() / (observed.sum() * values[observed])
        )
        if observed.any():
            raw[observed] /= raw[observed].mean()
        raw = np.clip(raw, 0.25, 4.0)
        raw[~observed] = 0.0
        weights[name] = torch.tensor(
            raw, dtype=torch.float32, device=context.device
        )
        report[name] = {"counts": values.tolist(), "weights": raw.tolist()}
    return weights, report


def _training_coverage_audit(context: RuntimeContext) -> dict[str, Any]:
    """Fail if capped training silently removes an episode's target event."""

    normal_index = context.targets.normal_index

    def episode_sets(training: bool) -> tuple[set[str], set[str], int, int]:
        episodes: set[str] = set()
        positive_episodes: set[str] = set()
        rows = positive_rows = 0
        for batch in _loader(
            _dataset(context, ("train_id",), training=training), context
        ):
            # Context remains available to the causal encoder, but loss/audit
            # exposure is restricted to each window's disjoint eval region so
            # a source row cannot receive duplicate loss merely due to overlap.
            selected = batch["valid_mask"] & batch["eval_mask"]
            selected_cpu = selected.detach().cpu()
            target = batch["class_target"]
            rows += int(selected_cpu.sum())
            positive_rows += int(((target != normal_index) & selected).sum())
            for index, episode_id in enumerate(batch["episode_id"]):
                if not bool(selected_cpu[index].any()):
                    continue
                episode = str(episode_id)
                episodes.add(episode)
                if bool(((target[index] != normal_index) & selected[index]).any()):
                    positive_episodes.add(episode)
        return episodes, positive_episodes, rows, positive_rows

    all_episodes, all_positive, all_rows, all_positive_rows = episode_sets(False)
    sampled_episodes, sampled_positive, sampled_rows, sampled_positive_rows = episode_sets(
        True
    )
    missing_episodes = sorted(all_episodes - sampled_episodes)
    missed_positive = sorted(all_positive - sampled_positive)
    if missing_episodes:
        raise ValueError(
            "training-window coverage audit omitted complete episodes: "
            f"{missing_episodes[:10]}"
        )
    if missed_positive:
        raise ValueError(
            "event-aware training-window coverage audit failed; sampled windows "
            f"miss non-N00 targets in {len(missed_positive)} episodes: "
            f"{missed_positive[:10]}"
        )
    return {
        "status": "pass",
        "normal_target_index": normal_index,
        "exhaustive_train_episodes": len(all_episodes),
        "sampled_train_episodes": len(sampled_episodes),
        "exhaustive_positive_target_episodes": len(all_positive),
        "sampled_positive_target_episodes": len(sampled_positive),
        "missed_positive_target_episodes": 0,
        "exhaustive_eval_rows": all_rows,
        "sampled_unique_optimizer_rows": sampled_rows,
        "optimizer_context_policy": (
            "overlap is encoder context only; loss uses valid_mask & eval_mask"
        ),
        "sampled_row_retention_fraction": sampled_rows / all_rows,
        "sampled_positive_row_retention_fraction": (
            sampled_positive_rows / all_positive_rows
            if all_positive_rows
            else None
        ),
        "exhaustive_positive_target_rows": all_positive_rows,
        "sampled_positive_target_rows": sampled_positive_rows,
        "sampler_policy": (
            "deterministic event-aware plus timeline-spanning; train labels only"
        ),
    }


def _critical_delay_coverage_audit(
    context: RuntimeContext,
    candidate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit usable local context for every critical exact-FIFO edge.

    Only ``train_id`` windows are constructed.  Coverage is evaluated over
    each window's disjoint optimisation/evaluation rows, with actual measured
    flow, profile-overlay edge volume, actual ``dt_s``, and the observable
    route gate used by the model.  A pooled value cannot hide an uncovered
    holding-tube edge.
    """

    gate_contract = candidate_contract["critical_delay_gate"]
    if gate_contract.get("fit_split") != "train_id":
        raise ValueError("critical-delay audit is restricted to train_id")
    edge_tags = tuple(map(str, gate_contract["edge_tags"]))
    graph_by_tag = {edge.tag: (index, edge) for index, edge in enumerate(context.graph.edges)}
    if set(edge_tags) - set(graph_by_tag):
        raise ValueError("critical-delay audit references a missing P&ID edge")
    for tag in edge_tags:
        _index, edge = graph_by_tag[tag]
        if edge.delay_kind != "exact_fifo" or not edge.advective:
            raise ValueError(
                f"critical edge {tag} must remain an advective exact_fifo edge"
            )

    route_gate = RouteGate(context.graph)
    flow_index = CONTROL_FIELDS.index("measured_flow_l_h")

    def scope(training: bool) -> dict[str, Any]:
        eligible_counts = {tag: 0 for tag in edge_tags}
        valid_counts = {tag: 0 for tag in edge_tags}
        window_count = 0
        for batch in _loader(
            _dataset(context, ("train_id",), training=training), context
        ):
            controls = batch["controls_raw"]
            dt_s = batch["dt_s"]
            volume = batch["edge_volume_l"]
            selected = batch["valid_mask"] & batch["eval_mask"]
            flow = controls[..., flow_index]
            gate = route_gate(controls)
            local_index = torch.arange(
                controls.shape[1], dtype=dt_s.dtype
            ).view(1, -1)
            for tag in edge_tags:
                edge_index = graph_by_tag[tag][0]
                delay_steps = (
                    volume[:, None, edge_index]
                    * 3600.0
                    / flow.clamp_min(1.0)
                    / dt_s
                )
                eligible = (
                    selected
                    & (flow > 1.0)
                    & (gate[..., edge_index] > 0.0)
                )
                delay_valid = local_index - delay_steps >= 0.0
                eligible_counts[tag] += int(eligible.sum())
                valid_counts[tag] += int((eligible & delay_valid).sum())
            window_count += int(controls.shape[0])
        rows: dict[str, Any] = {}
        for tag in edge_tags:
            denominator = eligible_counts[tag]
            rows[tag] = {
                "eligible_rows": denominator,
                "delay_valid_rows": valid_counts[tag],
                "coverage": (
                    valid_counts[tag] / denominator if denominator else None
                ),
            }
        coverages = [
            float(rows[tag]["coverage"])
            for tag in edge_tags
            if rows[tag]["coverage"] is not None
        ]
        return {
            "windows": window_count,
            "edges": rows,
            "minimum_per_edge_coverage": min(coverages) if coverages else None,
            "pooled_coverage": (
                sum(valid_counts.values()) / sum(eligible_counts.values())
                if sum(eligible_counts.values())
                else None
            ),
            "zero_denominator_edges": [
                tag for tag in edge_tags if eligible_counts[tag] == 0
            ],
        }

    exhaustive = scope(False)
    sampled = scope(True)
    exhaustive_limit = float(
        gate_contract["minimum_exhaustive_per_edge_coverage"]
    )
    sampled_limit = float(
        gate_contract["minimum_sampled_train_per_edge_coverage"]
    )
    exhaustive_minimum = exhaustive["minimum_per_edge_coverage"]
    sampled_minimum = sampled["minimum_per_edge_coverage"]
    passed = bool(
        not exhaustive["zero_denominator_edges"]
        and not sampled["zero_denominator_edges"]
        and exhaustive_minimum is not None
        and sampled_minimum is not None
        and exhaustive_minimum >= exhaustive_limit
        and sampled_minimum >= sampled_limit
    )
    return {
        "status": "pass" if passed else "fail",
        "fit_split": "train_id",
        "opened_splits": ["train_id"],
        "non_train_rows_opened": 0,
        "window_size": int(context.budget["window_size"]),
        "stride": int(context.budget["stride"]),
        "sampled_windows_per_episode": int(
            context.budget["train_windows_per_episode"]
        ),
        "edge_tags": list(edge_tags),
        "coverage_definition": gate_contract["valid_rows"],
        "exhaustive_train": exhaustive,
        "sampled_train": sampled,
        "thresholds": {
            "minimum_exhaustive_per_edge_coverage": exhaustive_limit,
            "minimum_sampled_train_per_edge_coverage": sampled_limit,
        },
        "pooled_values_are_decision_criteria": False,
        "post_hoc_development_gate": True,
        "post_hoc_design_disclosure": gate_contract["post_hoc_design_disclosure"],
    }


def _model_configs(
    context: RuntimeContext, variant: Variant
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Construct one variant plus its effective train/evaluation protocol."""

    budget = context.budget
    uncertainty = torch.tensor(
        normalized_sensor_uncertainty(context.graph, context.standardizer),
        dtype=torch.float32,
    )
    dimensions = {
        "feature_count": len(context.graph.features),
        "node_count": len(context.graph.nodes),
        "edge_count": len(context.graph.edges),
        "class_count": len(context.targets.codes),
        "location_count": len(context.targets.locations),
        "mechanism_count": len(context.targets.mechanisms),
    }
    common = {
        **dimensions,
        "hidden_dim": int(budget["hidden_dim"]),
        "dropout": float(budget["dropout"]),
        "observer_hidden_dim": int(budget["observer_hidden_dim"]),
    }
    if variant.model_name == V03_CANDIDATE_NAME:
        if context.targets.normal_index != 0:
            raise ValueError("v0.3 candidate requires N00 at taxonomy index zero")
        fault_indices = tuple(
            index
            for index, is_fault in enumerate(context.targets.code_is_fault)
            if is_fault
        )
        hybrid_config = FlowTwinHybridConfig(
            **dimensions,
            fault_class_indices=fault_indices,
            hidden_dim=int(budget["hidden_dim"]),
            temporal_layers=int(budget["layers"]),
            graph_layers=int(budget["layers"]),
            kernel_size=3,
            dropout=float(budget["dropout"]),
            observer_hidden_dim=int(budget["observer_hidden_dim"]),
        )
        model = FlowTwinHybrid(context.graph, hybrid_config, uncertainty)
        if tuple(model.config.fault_class_indices) != fault_indices:
            raise ValueError("hybrid checkpoint taxonomy fault partition mismatch")
        training_options = {
            "train_nominal_observer": True,
            "requires_counterfactual_reference": True,
            "loss_weight_overrides": {
                "counterfactual": 0.1,
                "delay_regularization": 0.001,
            },
        }
        evaluation_options = {
            "conformal_enabled": True,
            "abstention_enabled": True,
            "decision_policy": "validation_only_conformal",
            "probability_calibration": (
                "taxonomy_hierarchical_binary_and_conditional_temperature"
            ),
            "fault_class_indices": list(fault_indices),
        }
        serialized = {
            "kind": "flowtwin_v03_experimental_candidate",
            "model": hybrid_config.to_dict(),
            "training_options": training_options,
            "evaluation_options": evaluation_options,
            "ood_score_definition": (
                "log1p(mean normalized observer residual squared) + "
                "log1p(route-active transport error); higher means less nominal"
            ),
            "ood_score_comparability": (
                "not numerically interchangeable with FlowTwin v0.2 energy"
            ),
        }
    elif variant.model_name == FULL_MODEL_NAME:
        model_config = FlowTwinConfig(
            **common,
            graph_layers=int(budget["layers"]),
        )
        ablation = variant.ablation or "none"
        model = build_ablation_model(
            context.graph,
            model_config,
            uncertainty,
            ablation,
        )
        training_options = dict(model.training_options)
        evaluation_options = dict(model.evaluation_options)
        serialized = {
            "kind": "flowtwin_guard",
            "model": model_config.to_dict(),
            "ablation": model.ablation_config.to_dict(),
            "training_options": training_options,
            "evaluation_options": evaluation_options,
        }
    elif variant.model_name == DSPR_DIAGNOSTIC_ADAPTATION_NAME:
        dspr_config = DSPRDiagnosticConfig(
            **dimensions,
            hidden_dim=int(budget["hidden_dim"]),
            attention_heads=int(budget["attention_heads"]),
            dropout=float(budget["dropout"]),
        )
        model = build_dspr_diagnostic_adaptation(context.graph, dspr_config)
        training_options = {
            "train_nominal_observer": False,
            "requires_counterfactual_reference": False,
            "loss_weight_overrides": {
                "counterfactual": 0.0,
                "delay_regularization": 0.0,
            },
        }
        evaluation_options = {
            "conformal_enabled": True,
            "abstention_enabled": True,
            "decision_policy": "validation_only_conformal",
        }
        serialized = {
            "kind": "closest_prior_art_diagnostic_adaptation",
            "model": dspr_config.to_dict(),
            "training_options": training_options,
            "evaluation_options": evaluation_options,
        }
    else:
        baseline_config = BaselineConfig(
            **common,
            model_name=variant.model_name,
            layers=int(budget["layers"]),
            kernel_size=3,
            attention_heads=int(budget["attention_heads"]),
        )
        model = build_baseline(context.graph, baseline_config, uncertainty)
        has_observer = variant.model_name == "twin_residual_tcn"
        training_options = {
            "train_nominal_observer": has_observer,
            "requires_counterfactual_reference": False,
            "loss_weight_overrides": {
                "counterfactual": 0.0,
                "delay_regularization": 0.0,
            },
        }
        evaluation_options = {
            "conformal_enabled": True,
            "abstention_enabled": True,
            "decision_policy": "validation_only_conformal",
        }
        serialized = {
            "kind": "neural_baseline",
            "model": baseline_config.to_dict(),
            "training_options": training_options,
            "evaluation_options": evaluation_options,
        }
    if context.candidate_contract is not None:
        evaluation_options["operational_alarm"] = dict(
            context.candidate_contract["alarm_policy"]
        )
    model = model.to(context.device)
    return model, serialized, {
        "training": training_options,
        "evaluation": evaluation_options,
    }


def _observer_pretrain(
    model: nn.Module,
    context: RuntimeContext,
    training_options: Mapping[str, Any],
) -> list[dict[str, float]]:
    if not bool(training_options["train_nominal_observer"]):
        if hasattr(model, "set_observer_trainable"):
            model.set_observer_trainable(False)
        return []
    if not all(
        hasattr(model, name)
        for name in ("observer", "observer_outputs", "observer_nll")
    ):
        raise ValueError("variant declares observer pretraining but lacks observer API")
    model.set_observer_trainable(True)
    optimizer = torch.optim.AdamW(
        model.observer.parameters(),
        lr=float(context.budget["learning_rate"]),
        weight_decay=float(context.budget["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(context.budget["observer_epochs"])):
        model.observer.train()
        loss_total = 0.0
        batches = 0
        for raw_batch in _loader(
            _dataset(
                context,
                ("train_id",),
                training=True,
                normal_only=True,
            ),
            context,
        ):
            batch = _move_batch(raw_batch, context.device)
            optimizer.zero_grad(set_to_none=True)
            mean, log_variance, residual, observer_valid = model.observer_outputs(
                batch["x"], batch["valid_mask"]
            )
            output = {
                "nominal_mean": mean,
                "nominal_log_variance": log_variance,
                "residual": residual,
                "observer_valid": observer_valid,
            }
            observer_loss_mask = batch["valid_mask"] & batch["eval_mask"]
            loss = model.observer_nll(batch["x"], output, observer_loss_mask)
            if not torch.isfinite(loss):
                raise ValueError("non-finite nominal-observer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.observer.parameters(),
                float(context.budget["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("nominal observer received no train_id windows")
        history.append({"epoch": epoch + 1, "loss": loss_total / batches})
    model.set_observer_trainable(False)
    model.observer.eval()
    return history


def _zero_loss(output: Mapping[str, Tensor]) -> Tensor:
    return output["class_logits"].sum() * 0.0


def _transport_loss(
    output: Mapping[str, Tensor],
    class_target: Tensor,
    valid_mask: Tensor,
    normal_index: int,
) -> Tensor:
    if "transport_error" not in output:
        return _zero_loss(output)
    selected = valid_mask & (class_target == normal_index)
    if not torch.any(selected):
        return output["transport_error"].sum() * 0.0
    return output["transport_error"][selected].mean()


def _diagnostic_train(
    model: nn.Module,
    context: RuntimeContext,
    target_weights: Mapping[str, Tensor],
    training_options: Mapping[str, Any],
    registered_loss_weights: Mapping[str, float],
) -> list[dict[str, float]]:
    loss_weights = {name: float(value) for name, value in registered_loss_weights.items()}
    loss_weights.update(
        {
            name: float(value)
            for name, value in training_options["loss_weight_overrides"].items()
        }
    )
    if hasattr(model, "ablation_config"):
        model.ablation_config.assert_training_protocol(
            train_nominal_observer=bool(training_options["train_nominal_observer"]),
            counterfactual_loss_weight=loss_weights["counterfactual"],
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(context.budget["learning_rate"]),
        weight_decay=float(context.budget["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(context.budget["diagnostic_epochs"])):
        model.train()
        if hasattr(model, "observer"):
            model.observer.eval()
        totals: Counter[str] = Counter()
        batches = 0
        for raw_batch in _loader(
            _dataset(context, ("train_id",), training=True), context
        ):
            batch = _move_batch(raw_batch, context.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["x"],
                batch["controls_raw"],
                batch["dt_s"],
                batch["edge_volume_l"],
                batch["valid_mask"],
            )
            valid = batch["valid_mask"] & batch["eval_mask"]
            losses = {
                "class": focal_cross_entropy(
                    output["class_logits"],
                    batch["class_target"],
                    valid,
                    class_weight=target_weights["class"],
                ),
                "anomaly": focal_cross_entropy(
                    output["anomaly_logits"],
                    batch["anomaly_target"],
                    valid,
                    class_weight=target_weights["anomaly"],
                ),
                "location": focal_cross_entropy(
                    output["location_logits"],
                    batch["location_target"],
                    valid,
                    class_weight=target_weights["location"],
                ),
                "mechanism": focal_cross_entropy(
                    output["mechanism_logits"],
                    batch["mechanism_target"],
                    valid,
                    class_weight=target_weights["mechanism"],
                ),
                "transport": _transport_loss(
                    output,
                    batch["class_target"],
                    valid,
                    context.targets.normal_index,
                ),
            }
            if loss_weights["counterfactual"] > 0.0:
                reference = model(
                    batch["reference_x"],
                    batch["reference_controls_raw"],
                    batch["dt_s"],
                    batch["edge_volume_l"],
                    batch["valid_mask"],
                )
                losses["counterfactual"] = counterfactual_contrastive_loss(
                    output["embedding"],
                    reference["embedding"],
                    batch["class_target"],
                    valid,
                    normal_index=context.targets.normal_index,
                )
            else:
                losses["counterfactual"] = _zero_loss(output)
            losses["delay_regularization"] = (
                model.delay_regularization()
                if hasattr(model, "delay_regularization")
                else _zero_loss(output)
            )
            losses["auxiliary"] = (
                model.auxiliary_loss()
                if hasattr(model, "auxiliary_loss")
                else _zero_loss(output)
            )
            loss = sum(loss_weights[name] * value for name, value in losses.items())
            if not torch.isfinite(loss):
                raise ValueError("non-finite diagnostic training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(context.budget["gradient_clip_norm"])
            )
            optimizer.step()
            totals["total"] += float(loss.detach().cpu())
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("diagnostic model received no train_id windows")
        history.append(
            {
                "epoch": epoch + 1,
                **{name: value / batches for name, value in totals.items()},
            }
        )
    return history


def _concat(chunks: list[np.ndarray], dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if not chunks:
        raise ValueError("evaluation produced no rows")
    result = np.concatenate(chunks, axis=0)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _collect(
    model: nn.Module,
    context: RuntimeContext,
    accepted_splits: Sequence[str],
) -> tuple[dict[str, Any], float]:
    """Collect each selected row once without retaining Python row dictionaries."""

    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "probabilities",
            "anomaly_probability",
            "energies",
            "class_target",
            "anomaly_target",
            "location_target",
            "location_prediction",
            "mechanism_target",
            "mechanism_prediction",
            "mode",
            "time_s",
            "step_dt_s",
            "effect_time_s",
            "detection_eligible",
            "fault_active",
            "safety_event",
            "unsafe_forward_l",
            "episode_index",
            "counterfactual_group_index",
            "profile_index",
            "split_index",
        )
    }
    episode_indexes: dict[str, int] = {}
    group_indexes: dict[str, int] = {}
    profile_indexes: dict[str, int] = {}
    split_indexes: dict[str, int] = {}
    cip_index = CONTROL_FIELDS.index("cip_cycle_active")
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for raw_batch in _loader(
            _dataset(context, accepted_splits, training=False), context
        ):
            batch = _move_batch(raw_batch, context.device)
            output = model(
                batch["x"],
                batch["controls_raw"],
                batch["dt_s"],
                batch["edge_volume_l"],
                batch["valid_mask"],
            )
            selected = batch["valid_mask"] & batch["eval_mask"]
            class_probability = torch.softmax(output["class_logits"], dim=-1)
            anomaly_probability = torch.softmax(
                output["anomaly_logits"], dim=-1
            )[..., 1]
            chunks["probabilities"].append(
                class_probability[selected].detach().cpu().numpy().astype(np.float32)
            )
            chunks["anomaly_probability"].append(
                anomaly_probability[selected].detach().cpu().numpy().astype(np.float32)
            )
            chunks["energies"].append(
                output["ood_energy"][selected].detach().cpu().numpy().astype(np.float32)
            )
            for name in (
                "class_target",
                "anomaly_target",
                "location_target",
                "mechanism_target",
            ):
                chunks[name].append(
                    batch[name][selected].detach().cpu().numpy().astype(np.int16)
                )
            chunks["location_prediction"].append(
                output["location_logits"][selected]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
            chunks["mechanism_prediction"].append(
                output["mechanism_logits"][selected]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )
            chunks["mode"].append(
                batch["controls_raw"][..., cip_index][selected]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            for name in (
                "time_s",
                "dt_s",
                "fault_active",
                "safety_event",
                "unsafe_forward_l",
            ):
                output_name = "step_dt_s" if name == "dt_s" else name
                chunks[output_name].append(
                    batch[name][selected].detach().cpu().numpy().astype(np.float32)
                )

            per_episode: list[np.ndarray] = []
            per_profile: list[np.ndarray] = []
            per_group: list[np.ndarray] = []
            per_split: list[np.ndarray] = []
            per_effect: list[np.ndarray] = []
            per_eligible: list[np.ndarray] = []
            selected_cpu = selected.detach().cpu()
            for batch_index in range(selected.shape[0]):
                count = int(selected_cpu[batch_index].sum())
                episode_id = str(raw_batch["episode_id"][batch_index])
                profile_id = str(raw_batch["plant_profile_id"][batch_index])
                group_id = str(
                    raw_batch["counterfactual_group_id"][batch_index]
                )
                split = str(raw_batch["split"][batch_index])
                episode_number = episode_indexes.setdefault(
                    episode_id, len(episode_indexes)
                )
                split_number = split_indexes.setdefault(split, len(split_indexes))
                profile_number = profile_indexes.setdefault(
                    profile_id, len(profile_indexes)
                )
                group_number = group_indexes.setdefault(group_id, len(group_indexes))
                effect = float(raw_batch["effect_time_s"][batch_index])
                eligible = bool(int(raw_batch["detection_eligible"][batch_index]))
                per_episode.append(np.full(count, episode_number, dtype=np.int32))
                per_profile.append(np.full(count, profile_number, dtype=np.int16))
                per_group.append(np.full(count, group_number, dtype=np.int32))
                per_split.append(np.full(count, split_number, dtype=np.int16))
                per_effect.append(
                    np.full(count, effect if eligible else np.nan, dtype=np.float32)
                )
                per_eligible.append(np.full(count, eligible, dtype=np.bool_))
            chunks["episode_index"].append(np.concatenate(per_episode))
            chunks["profile_index"].append(np.concatenate(per_profile))
            chunks["counterfactual_group_index"].append(np.concatenate(per_group))
            chunks["split_index"].append(np.concatenate(per_split))
            chunks["effect_time_s"].append(np.concatenate(per_effect))
            chunks["detection_eligible"].append(np.concatenate(per_eligible))
    elapsed = time.perf_counter() - started
    collected = {name: _concat(values) for name, values in chunks.items()}
    collected["episode_vocabulary"] = [
        name for name, _index in sorted(episode_indexes.items(), key=lambda item: item[1])
    ]
    collected["profile_vocabulary"] = [
        name for name, _index in sorted(profile_indexes.items(), key=lambda item: item[1])
    ]
    collected["counterfactual_group_vocabulary"] = [
        name for name, _index in sorted(group_indexes.items(), key=lambda item: item[1])
    ]
    collected["split_vocabulary"] = [
        name for name, _index in sorted(split_indexes.items(), key=lambda item: item[1])
    ]
    rows = len(collected["class_target"])
    return collected, elapsed / max(rows, 1)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    log_probability = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probability -= log_probability.max(axis=1, keepdims=True)
    scaled = np.exp(log_probability)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    if len(targets) == 0:
        raise ValueError("temperature calibration requires at least one row")
    candidates = np.exp(np.linspace(math.log(0.05), math.log(5.0), 241))
    indexes = np.arange(len(targets))
    best_temperature = 1.0
    best_nll = math.inf
    for candidate in candidates:
        scaled = _temperature_scale(probabilities, float(candidate))
        nll = float(
            -np.log(np.clip(scaled[indexes, targets], 1e-12, 1.0)).mean()
        )
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(candidate)
    return best_temperature


def _taxonomy_hierarchical_scale(
    probabilities: np.ndarray,
    anomaly_probability: np.ndarray,
    fault_class_indices: Sequence[int],
    *,
    anomaly_temperature: float,
    non_fault_temperature: float,
    fault_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale binary and conditional probabilities without breaking taxonomy."""

    probability = np.asarray(probabilities, dtype=np.float64)
    anomaly = np.asarray(anomaly_probability, dtype=np.float64)
    if probability.ndim != 2 or anomaly.shape != (len(probability),):
        raise ValueError("hierarchical calibration probability shapes differ")
    fault = tuple(map(int, fault_class_indices))
    if not fault or len(set(fault)) != len(fault):
        raise ValueError("hierarchical calibration needs unique fault classes")
    if min(fault) < 0 or max(fault) >= probability.shape[1]:
        raise ValueError("hierarchical fault class index is out of range")
    non_fault = tuple(index for index in range(probability.shape[1]) if index not in fault)
    if not non_fault:
        raise ValueError("hierarchical calibration needs non-fault classes")

    binary = np.column_stack([1.0 - anomaly, anomaly])
    binary_scaled = _temperature_scale(binary, anomaly_temperature)
    non_fault_probability = probability[:, non_fault]
    non_fault_probability /= non_fault_probability.sum(axis=1, keepdims=True).clip(
        1e-12
    )
    fault_probability = probability[:, fault]
    fault_probability /= fault_probability.sum(axis=1, keepdims=True).clip(1e-12)
    non_fault_scaled = _temperature_scale(
        non_fault_probability, non_fault_temperature
    )
    fault_scaled = _temperature_scale(fault_probability, fault_temperature)
    scaled = np.zeros_like(probability)
    scaled[:, non_fault] = binary_scaled[:, :1] * non_fault_scaled
    scaled[:, fault] = binary_scaled[:, 1:] * fault_scaled
    if not np.allclose(scaled.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("hierarchical calibration failed probability normalization")
    return scaled, binary_scaled[:, 1]


def _fit_taxonomy_hierarchical_calibration(
    validation: Mapping[str, Any],
    fault_class_indices: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    probability = np.asarray(validation["probabilities"], dtype=np.float64)
    anomaly = np.asarray(validation["anomaly_probability"], dtype=np.float64)
    class_target = np.asarray(validation["class_target"], dtype=np.int64)
    anomaly_target = np.asarray(validation["anomaly_target"], dtype=np.int64)
    fault = tuple(map(int, fault_class_indices))
    non_fault = tuple(index for index in range(probability.shape[1]) if index not in fault)
    observed_fault_mass = probability[:, fault].sum(axis=1)
    raw_error = float(np.max(np.abs(observed_fault_mass - anomaly)))
    if raw_error > 1e-5:
        raise ValueError(
            "hybrid class/anomaly hierarchy is incoherent before calibration: "
            f"max error {raw_error}"
        )
    anomaly_temperature = _fit_temperature(
        np.column_stack([1.0 - anomaly, anomaly]), anomaly_target
    )

    def conditional_temperature(indices: tuple[int, ...], selected: np.ndarray) -> float:
        if not np.any(selected):
            raise ValueError("hierarchical calibration lacks one taxonomy partition")
        conditional = probability[selected][:, indices]
        conditional /= conditional.sum(axis=1, keepdims=True).clip(1e-12)
        index_map = {class_index: position for position, class_index in enumerate(indices)}
        targets = np.asarray(
            [index_map[int(value)] for value in class_target[selected]],
            dtype=np.int64,
        )
        return _fit_temperature(conditional, targets)

    is_fault = np.isin(class_target, fault)
    fault_temperature = conditional_temperature(fault, is_fault)
    non_fault_temperature = conditional_temperature(non_fault, ~is_fault)
    scaled, anomaly_scaled = _taxonomy_hierarchical_scale(
        probability,
        anomaly,
        fault,
        anomaly_temperature=anomaly_temperature,
        non_fault_temperature=non_fault_temperature,
        fault_temperature=fault_temperature,
    )
    calibrated_error = float(
        np.max(np.abs(scaled[:, fault].sum(axis=1) - anomaly_scaled))
    )
    state = {
        "method": "validation-only taxonomy-hierarchical temperature scaling",
        "fault_class_indices": list(fault),
        "non_fault_class_indices": list(non_fault),
        "anomaly_temperature": anomaly_temperature,
        "non_fault_conditional_temperature": non_fault_temperature,
        "fault_conditional_temperature": fault_temperature,
        "raw_max_fault_mass_error": raw_error,
        "calibrated_max_fault_mass_error": calibrated_error,
        "coherence_invariant": (
            "sum calibrated fault-class probability == calibrated anomaly probability"
        ),
    }
    return state, scaled.astype(np.float32), anomaly_scaled.astype(np.float32)


def _binary_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def _fit_anomaly_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = _binary_f1(truth, probability >= threshold)
        if f1 > best_f1 + 1e-12 or (
            abs(f1 - best_f1) <= 1e-12 and threshold > best_threshold
        ):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def _counterfactual_group_mode_calibration(
    validation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reduce correlated validation rows to conservative group/mode blocks."""

    group_names = np.asarray(validation["counterfactual_group_vocabulary"], dtype=object)[
        np.asarray(validation["counterfactual_group_index"], dtype=np.int64)
    ]
    profile_names = np.asarray(validation["profile_vocabulary"], dtype=object)[
        np.asarray(validation["profile_index"], dtype=np.int64)
    ]
    return counterfactual_group_mode_block_max(
        validation["probabilities"],
        validation["class_target"],
        validation["mode"],
        validation["energies"],
        group_names,
        profile_names,
    )


def _fit_validation_calibration(
    validation: dict[str, Any],
    context: RuntimeContext,
    evaluation_options: Mapping[str, Any],
) -> tuple[dict[str, Any], ModeConformalCalibrator | None]:
    """Fit every learned post-processing value using validation_id only."""

    if validation["split_vocabulary"] != ["validation_id"]:
        raise ValueError(
            "calibration firewall violation: inputs must contain validation_id only"
        )
    hierarchical = evaluation_options.get("probability_calibration") == (
        "taxonomy_hierarchical_binary_and_conditional_temperature"
    )
    hierarchy_state: dict[str, Any] | None = None
    if hierarchical:
        hierarchy_state, scaled_probability, scaled_anomaly = (
            _fit_taxonomy_hierarchical_calibration(
                validation,
                evaluation_options["fault_class_indices"],
            )
        )
        validation["probabilities"] = scaled_probability
        validation["anomaly_probability"] = scaled_anomaly
        temperature = 1.0
        temperature_method = (
            "not_applicable; see taxonomy_hierarchical_calibration"
        )
    else:
        temperature = _fit_temperature(
            validation["probabilities"], validation["class_target"]
        )
        validation["probabilities"] = _temperature_scale(
            validation["probabilities"], temperature
        ).astype(np.float32)
        temperature_method = "deterministic validation NLL grid search"
    anomaly_threshold = _fit_anomaly_threshold(
        validation["anomaly_probability"], validation["anomaly_target"]
    )
    conformal_enabled = bool(evaluation_options["conformal_enabled"])
    calibrator: ModeConformalCalibrator | None = None
    state: dict[str, Any] = {
        "fit_split": "validation_id",
        "test_rows_seen_during_fit": 0,
        "class_temperature": temperature,
        "class_temperature_method": temperature_method,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_threshold_method": (
            "maximum validation row-F1; highest-threshold tie break"
        ),
        "conformal_enabled": conformal_enabled,
    }
    if hierarchy_state is not None:
        state["taxonomy_hierarchical_calibration"] = hierarchy_state
    if conformal_enabled:
        block_calibration, block_audit = _counterfactual_group_mode_calibration(
            validation
        )
        calibrator = ModeConformalCalibrator(
            alpha=float(context.budget["alpha"]),
            ood_alpha=float(context.budget["ood_alpha"]),
        ).fit(
            block_calibration["probabilities"],
            block_calibration["targets"],
            block_calibration["modes"],
            block_calibration["energies"],
            splits=["validation_id"] * len(block_calibration["targets"]),
        )
        state["conformal"] = calibrator.to_dict()
        state["conformal_calibration_audit"] = block_audit
    else:
        state["conformal"] = {
            "status": "not_applicable",
            "reason": "registered no_conformal ablation",
        }
    alarm_contract = evaluation_options.get("operational_alarm")
    if alarm_contract is not None:
        episode_ids = np.asarray(
            validation["episode_vocabulary"], dtype=object
        )[np.asarray(validation["episode_index"], dtype=np.int64)]
        profile_ids = np.asarray(
            validation["profile_vocabulary"], dtype=object
        )[np.asarray(validation["profile_index"], dtype=np.int64)]
        split_ids = np.asarray(
            validation["split_vocabulary"], dtype=object
        )[np.asarray(validation["split_index"], dtype=np.int64)]
        alarm_kwargs: dict[str, Any] = {
            "scores": validation["anomaly_probability"],
            "truth_active": validation["anomaly_target"],
            "time_s": validation["time_s"],
            "step_dt_s": validation["step_dt_s"],
            "effect_time_s": validation["effect_time_s"],
            "episode_ids": episode_ids,
            "profile_ids": profile_ids,
            "splits": split_ids,
            "detection_eligible": validation["detection_eligible"],
            "on_thresholds": alarm_contract["on_thresholds"],
            "off_thresholds": alarm_contract["off_thresholds"],
            "min_on_durations_s": alarm_contract["min_on_durations_s"],
            "min_off_durations_s": alarm_contract["min_off_durations_s"],
            "cooldowns_s": alarm_contract["cooldowns_s"],
            "max_false_alarm_onsets_per_negative_hour": alarm_contract[
                "maximum_profile_false_alarm_onsets_per_negative_hour"
            ],
            "objective": alarm_contract["objective"],
            "min_event_recall": alarm_contract.get("minimum_event_recall"),
            "min_profile_event_recall": alarm_contract.get(
                "minimum_profile_event_recall"
            ),
        }
        state["operational_alarm"] = fit_alarm_policy(**alarm_kwargs)
    return state, calibrator


def _prediction_sets(
    probabilities: np.ndarray,
    modes: np.ndarray,
    energies: np.ndarray,
    calibrator: ModeConformalCalibrator,
) -> tuple[np.ndarray, np.ndarray]:
    calibrator._require_fit()
    assert calibrator.global_threshold is not None
    assert calibrator.mode_thresholds is not None
    assert calibrator.energy_threshold is not None
    production = calibrator.mode_thresholds.get(
        "production", calibrator.global_threshold
    )
    cip = calibrator.mode_thresholds.get("cip", calibrator.global_threshold)
    threshold = np.where(modes >= 0.5, cip, production)
    sets = (1.0 - probabilities) <= threshold[:, None]
    sizes = sets.sum(axis=1)
    decisions = np.full(len(probabilities), 2, dtype=np.int8)  # REVIEW
    decisions[sizes == 1] = 1  # DIAGNOSE
    decisions[(sizes == 0) | (energies > calibrator.energy_threshold)] = 0  # UNKNOWN
    return sets, decisions


def _unsafe_before_detection(
    collected: Mapping[str, Any],
    alarm: np.ndarray,
    selected: np.ndarray | None = None,
) -> float:
    total = 0.0
    episode = collected["episode_index"]
    included = (
        np.ones(len(episode), dtype=np.bool_)
        if selected is None
        else np.asarray(selected, dtype=np.bool_)
    )
    for episode_id in np.unique(episode[included]):
        episode_rows = np.flatnonzero((episode == episode_id) & included)
        order = episode_rows[
            np.argsort(collected["time_s"][episode_rows], kind="stable")
        ]
        eligible = collected["detection_eligible"][order].astype(bool)
        truth = collected["anomaly_target"][order].astype(bool)
        if not np.any(eligible & truth):
            continue
        effect_values = collected["effect_time_s"][order][eligible & truth]
        effect = float(effect_values[0])
        after = collected["time_s"][order] + 1e-9 >= effect
        detected = np.flatnonzero(alarm[order] & after & truth)
        cutoff = (
            float(collected["time_s"][order[detected[0]]])
            if detected.size
            else math.inf
        )
        include = after & (collected["time_s"][order] < cutoff)
        total += float(collected["unsafe_forward_l"][order][include].sum())
    return total


def _metric_scope(
    collected: Mapping[str, Any],
    selected: np.ndarray,
    targets: TargetContract,
    anomaly_threshold: float,
    prediction_sets: np.ndarray | None,
    decisions: np.ndarray | None,
    operational_alarm: np.ndarray | None = None,
) -> dict[str, Any]:
    truth = collected["class_target"][selected].astype(np.int64)
    probability = collected["probabilities"][selected]
    predicted = probability.argmax(axis=1)
    anomaly_truth = collected["anomaly_target"][selected].astype(np.int64)
    anomaly_alarm = (
        collected["anomaly_probability"][selected] >= anomaly_threshold
    )
    scoped_operational_alarm = (
        None
        if operational_alarm is None
        else np.asarray(operational_alarm, dtype=np.bool_)[selected]
    )
    location_truth = collected["location_target"][selected].astype(np.int64)
    mechanism_truth = collected["mechanism_target"][selected].astype(np.int64)
    episode = collected["episode_index"][selected]
    time_values = collected["time_s"][selected]
    step = collected["step_dt_s"][selected]
    effects = collected["effect_time_s"][selected]
    eligible = collected["detection_eligible"][selected]
    result: dict[str, Any] = {
        "rows": int(len(truth)),
        "class_diagnosis": classification_metrics(
            truth,
            predicted,
            labels=list(range(len(targets.codes))),
        ),
        "anomaly_rows": classification_metrics(
            anomaly_truth,
            anomaly_alarm.astype(np.int64),
            labels=[0, 1],
        ),
        "location_diagnosis": classification_metrics(
            location_truth,
            collected["location_prediction"][selected].astype(np.int64),
            labels=list(range(len(targets.locations))),
        ),
        "mechanism_diagnosis": classification_metrics(
            mechanism_truth,
            collected["mechanism_prediction"][selected].astype(np.int64),
            labels=list(range(len(targets.mechanisms))),
        ),
        "event_detection": event_detection_metrics(
            anomaly_truth,
            anomaly_alarm,
            time_values,
            step,
            effect_time_s=effects,
            episode_ids=episode,
            detection_eligible=eligible,
        ),
        "false_alarms": false_alarm_metrics(
            anomaly_truth,
            anomaly_alarm,
            time_values,
            step,
            episode_ids=episode,
        ),
        "process_volume": volume_metrics(
            collected["unsafe_forward_l"][selected], None, None
        ),
        "selective_risk": risk_coverage_curve(
            (truth != predicted).astype(np.float64),
            probability.max(axis=1),
            max_points=101,
        ),
        "unsafe_forward_l_before_first_post_effect_alarm": (
            _unsafe_before_detection(
                collected,
                collected["anomaly_probability"] >= anomaly_threshold,
                selected,
            )
        ),
    }
    for item, code in zip(
        result["class_diagnosis"]["per_class"], targets.codes, strict=True
    ):
        item["code"] = code
    for item, location in zip(
        result["location_diagnosis"]["per_class"],
        targets.locations,
        strict=True,
    ):
        item["location"] = location
    for item, mechanism in zip(
        result["mechanism_diagnosis"]["per_class"],
        targets.mechanisms,
        strict=True,
    ):
        item["mechanism"] = mechanism
    if prediction_sets is None:
        result["conformal"] = {
            "status": "not_applicable",
            "reason": "registered no_conformal ablation",
        }
        result["decisions"] = {
            "status": "not_applicable",
            "reason": "no abstention policy in no_conformal ablation",
        }
    else:
        scoped_sets = prediction_sets[selected]
        result["conformal"] = conformal_metrics(truth, scoped_sets)
        assert decisions is not None
        counts = Counter(int(value) for value in decisions[selected])
        result["decisions"] = {
            "UNKNOWN": counts[0],
            "DIAGNOSE": counts[1],
            "REVIEW": counts[2],
        }
    if scoped_operational_alarm is None:
        result["operational_alarm"] = {
            "status": "not_applicable",
            "reason": "no validation-fitted operational alarm contract",
        }
    else:
        result["operational_alarm"] = {
            "status": "reported",
            "active_rows": int(scoped_operational_alarm.sum()),
            "active_fraction": float(scoped_operational_alarm.mean()),
        }
        result["operational_event_detection"] = event_detection_metrics(
            anomaly_truth,
            scoped_operational_alarm,
            time_values,
            step,
            effect_time_s=effects,
            episode_ids=episode,
            detection_eligible=eligible,
        )
        result["operational_false_alarms"] = false_alarm_metrics(
            anomaly_truth,
            scoped_operational_alarm,
            time_values,
            step,
            episode_ids=episode,
        )
        result["operational_unsafe_forward_l_before_first_post_effect_alarm"] = (
            _unsafe_before_detection(collected, operational_alarm, selected)
        )
    return result


def _evaluate_metrics(
    collected: dict[str, Any],
    targets: TargetContract,
    calibration: Mapping[str, Any],
    calibrator: ModeConformalCalibrator | None,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    hierarchy = calibration.get("taxonomy_hierarchical_calibration")
    if hierarchy is None:
        temperature = float(calibration["class_temperature"])
        collected["probabilities"] = _temperature_scale(
            collected["probabilities"], temperature
        ).astype(np.float32)
    else:
        scaled_probability, scaled_anomaly = _taxonomy_hierarchical_scale(
            collected["probabilities"],
            collected["anomaly_probability"],
            hierarchy["fault_class_indices"],
            anomaly_temperature=float(hierarchy["anomaly_temperature"]),
            non_fault_temperature=float(
                hierarchy["non_fault_conditional_temperature"]
            ),
            fault_temperature=float(hierarchy["fault_conditional_temperature"]),
        )
        collected["probabilities"] = scaled_probability.astype(np.float32)
        collected["anomaly_probability"] = scaled_anomaly.astype(np.float32)
    prediction_sets: np.ndarray | None = None
    decisions: np.ndarray | None = None
    if calibrator is not None:
        prediction_sets, decisions = _prediction_sets(
            collected["probabilities"],
            collected["mode"],
            collected["energies"],
            calibrator,
        )
    anomaly_threshold = float(calibration["anomaly_threshold"])
    operational_alarm: np.ndarray | None = None
    alarm_state = calibration.get("operational_alarm")
    if alarm_state is not None:
        if alarm_state.get("status") != "selected" or alarm_state.get("config") is None:
            raise ValueError("operational alarm has no validation-selected operating point")
        operational_alarm = apply_alarm_policy(
            collected["anomaly_probability"],
            collected["step_dt_s"],
            collected["episode_index"],
            AlarmPolicyConfig.from_dict(alarm_state["config"]),
        )
        collected["operational_alarm"] = operational_alarm
    n = len(collected["class_target"])
    all_rows = np.ones(n, dtype=np.bool_)
    overall = _metric_scope(
        collected,
        all_rows,
        targets,
        anomaly_threshold,
        prediction_sets,
        decisions,
        operational_alarm,
    )
    by_split: dict[str, Any] = {}
    for split_index, name in enumerate(collected["split_vocabulary"]):
        selected = collected["split_index"] == split_index
        by_split[name] = _metric_scope(
            collected,
            selected,
            targets,
            anomaly_threshold,
            prediction_sets,
            decisions,
            operational_alarm,
        )
    by_profile: dict[str, Any] = {}
    profile_splits: dict[str, str] = {}
    for profile_index, name in enumerate(collected["profile_vocabulary"]):
        selected = collected["profile_index"] == profile_index
        split_indexes = np.unique(collected["split_index"][selected])
        if split_indexes.size != 1:
            raise ValueError(
                f"plant profile {name!r} spans multiple evaluation splits"
            )
        profile_splits[name] = str(
            collected["split_vocabulary"][int(split_indexes[0])]
        )
        by_profile[name] = _metric_scope(
            collected,
            selected,
            targets,
            anomaly_threshold,
            prediction_sets,
            decisions,
            operational_alarm,
        )

    is_ood = np.asarray(
        [
            str(collected["split_vocabulary"][int(index)]).startswith("test_ood")
            for index in collected["split_index"]
        ],
        dtype=np.bool_,
    )
    ood = ood_detection_metrics(is_ood, collected["energies"])
    if not ood["positives"] or not ood["negatives"]:
        ood.update(
            {
                "status": "not_applicable",
                "reason": (
                    "OOD metrics require both test_id and held-out-profile test "
                    "rows; no examples are fabricated or relabelled"
                ),
            }
        )
    else:
        ood["status"] = "reported"

    anomaly_alarm = collected["anomaly_probability"] >= anomaly_threshold
    result = {
        "metrics_version": METRICS_VERSION,
        "evaluation_scope": list(collected["split_vocabulary"]),
        "overall": overall,
        "by_split": by_split,
        "by_profile": by_profile,
        "profile_splits": profile_splits,
        "ood_detection": ood,
        "unsafe_forward_l_before_first_post_effect_alarm": _unsafe_before_detection(
            collected, anomaly_alarm
        ),
        "anomaly_threshold": anomaly_threshold,
        "anomaly_threshold_fit_split": "validation_id",
        "claim_boundary": (
            "Synthetic research benchmark only; predictions never authorize SAFE "
            "or RELEASE and never override PLC/HACCP logic."
        ),
    }
    if operational_alarm is not None:
        result["operational_alarm_policy"] = {
            "version": ALARM_POLICY_VERSION,
            "fit_split": "validation_id",
            "config": alarm_state["config"],
            "validation_audit": alarm_state["audit"],
        }
        result[
            "operational_unsafe_forward_l_before_first_post_effect_alarm"
        ] = _unsafe_before_detection(collected, operational_alarm)
    return result, prediction_sets, decisions


def _save_prediction_arrays(
    path: Path,
    collected: Mapping[str, Any],
    prediction_sets: np.ndarray | None,
    decisions: np.ndarray | None,
) -> None:
    arrays = {
        name: value
        for name, value in collected.items()
        if isinstance(value, np.ndarray)
    }
    for name in (
        "episode_vocabulary",
        "counterfactual_group_vocabulary",
        "profile_vocabulary",
        "split_vocabulary",
    ):
        arrays[name] = np.asarray(collected[name], dtype=np.str_)
    arrays["predicted_class"] = collected["probabilities"].argmax(axis=1)
    if prediction_sets is not None:
        arrays["prediction_sets"] = prediction_sets
    if decisions is not None:
        arrays["decision_code"] = decisions
    np.savez_compressed(path, **arrays)


def _run_id(variant: Variant, seed: int) -> str:
    safe = variant.variant_id.replace(":", "-").replace("/", "-")
    return f"{safe}__seed-{seed}"


def _run_variant(
    variant: Variant,
    seed: int,
    context: RuntimeContext,
    target_weights: Mapping[str, Tensor],
    contract: Mapping[str, Any],
    output: Path,
    *,
    save_predictions: bool,
) -> dict[str, Any]:
    """Train, reload, calibrate, then test exactly one matrix cell."""

    _seed_everything(seed)
    run_id = _run_id(variant, seed)
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True)
    model, serialized_model, protocol = _model_configs(context, variant)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_before = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    training_started = time.perf_counter()
    observer_history = _observer_pretrain(
        model, context, protocol["training"]
    )
    diagnostic_history = _diagnostic_train(
        model,
        context,
        target_weights,
        protocol["training"],
        contract["loss_weights"],
    )
    training_seconds = time.perf_counter() - training_started
    checkpoint = {
        "benchmark_runner_version": BENCHMARK_RUNNER_VERSION,
        "variant": variant.to_dict(),
        "seed": seed,
        "serialized_model": serialized_model,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, run_dir / "model.pt")
    write_json(
        run_dir / "training_history.json",
        {"observer": observer_history, "diagnostic": diagnostic_history},
    )

    # Metrics are deliberately generated from a strict reconstruction/reload.
    reloaded, reloaded_config, reloaded_protocol = _model_configs(context, variant)
    saved = torch.load(run_dir / "model.pt", map_location=context.device, weights_only=True)
    if saved["variant"] != variant.to_dict() or saved["seed"] != seed:
        raise ValueError("checkpoint variant/seed contract mismatch")
    if reloaded_config != saved["serialized_model"]:
        raise ValueError("checkpoint model configuration mismatch")
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    if variant.model_name == V03_CANDIDATE_NAME:
        expected_fault_indices = torch.tensor(
            [
                index
                for index, is_fault in enumerate(context.targets.code_is_fault)
                if is_fault
            ],
            dtype=torch.long,
            device=reloaded.fault_class_indices.device,
        )
        if not torch.equal(reloaded.fault_class_indices, expected_fault_indices):
            raise ValueError("reloaded hybrid taxonomy fault partition mismatch")
    if hasattr(reloaded, "set_observer_trainable"):
        reloaded.set_observer_trainable(False)
    reloaded.eval()

    # Firewall: finish every post-processing fit before constructing test data.
    validation, validation_seconds_per_row = _collect(
        reloaded, context, ("validation_id",)
    )
    calibration, calibrator = _fit_validation_calibration(
        validation, context, reloaded_protocol["evaluation"]
    )
    if hasattr(reloaded, "ablation_config"):
        reloaded.ablation_config.assert_evaluation_protocol(
            conformal_enabled=bool(reloaded_protocol["evaluation"]["conformal_enabled"])
        )
    write_json(run_dir / "calibration.json", calibration)
    alarm_state = calibration.get("operational_alarm")
    if alarm_state is not None and alarm_state.get("status") != "selected":
        raise ValueError(
            f"{run_id}: validation produced no operational alarm point; "
            "test iterator remains unopened and row-threshold fallback is forbidden"
        )

    test_splits = tuple(
        sorted(
            name
            for name in {
                row["split"]
                for row in read_csv_rows(context.splits / "episode_splits.csv")
            }
            if name.startswith("test_")
        )
    )
    test, test_seconds_per_row = _collect(reloaded, context, test_splits)
    metrics, prediction_sets, decisions = _evaluate_metrics(
        test, context.targets, calibration, calibrator
    )
    metrics["efficiency"] = {
        "parameter_count": total_parameters,
        "trainable_parameter_count_before_observer_freeze": trainable_before,
        "training_wall_time_s": training_seconds,
        "validation_inference_rows_per_s": 1.0 / validation_seconds_per_row,
        "test_inference_rows_per_s": 1.0 / test_seconds_per_row,
    }
    write_json(run_dir / "metrics.json", metrics)
    if save_predictions:
        _save_prediction_arrays(
            run_dir / "test_predictions.npz", test, prediction_sets, decisions
        )
    run_record = {
        "run_id": run_id,
        "variant": variant.to_dict(),
        "seed": seed,
        "model": serialized_model,
        "training_protocol": protocol["training"],
        "evaluation_protocol": protocol["evaluation"],
        "calibration_sha256": sha256_file(run_dir / "calibration.json"),
        "checkpoint_sha256": sha256_file(run_dir / "model.pt"),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "training_history_sha256": sha256_file(run_dir / "training_history.json"),
        "metrics": metrics,
    }
    write_json(run_dir / "run_record.json", run_record)
    return run_record


ENDPOINTS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "class_macro_f1": (
        "test_id",
        ("class_diagnosis", "macro_f1"),
        "higher_is_better",
    ),
    "event_f1": (
        "overall",
        ("event_detection", "event_f1"),
        "higher_is_better",
    ),
    "event_recall": (
        "overall",
        ("event_detection", "event_recall"),
        "higher_is_better",
    ),
    "horizon_penalized_detection_latency_mean_s": (
        "overall",
        ("event_detection", "horizon_penalized_latency_mean_s"),
        "lower_is_better",
    ),
    "false_alarm_onsets_per_negative_hour": (
        "overall",
        ("false_alarms", "false_alarm_onsets_per_negative_hour"),
        "lower_is_better",
    ),
    "unsafe_forward_l_before_detection": (
        "overall",
        ("unsafe_forward_l_before_first_post_effect_alarm",),
        "lower_is_better",
    ),
    "operational_event_f1": (
        "overall",
        ("operational_event_detection", "event_f1"),
        "higher_is_better",
    ),
    "operational_event_recall": (
        "overall",
        ("operational_event_detection", "event_recall"),
        "higher_is_better",
    ),
    "operational_horizon_penalized_detection_latency_mean_s": (
        "overall",
        ("operational_event_detection", "horizon_penalized_latency_mean_s"),
        "lower_is_better",
    ),
    "operational_false_alarm_onsets_per_negative_hour": (
        "overall",
        ("operational_false_alarms", "false_alarm_onsets_per_negative_hour"),
        "lower_is_better",
    ),
    "operational_unsafe_forward_l_before_detection": (
        "overall",
        ("operational_unsafe_forward_l_before_first_post_effect_alarm",),
        "lower_is_better",
    ),
    "conformal_coverage": (
        "overall",
        ("conformal", "empirical_coverage"),
        "target_is_one_minus_alpha",
    ),
    "mean_prediction_set_size": (
        "overall",
        ("conformal", "mean_prediction_set_size"),
        "lower_at_valid_coverage_is_better",
    ),
    "selective_aurc": (
        "overall",
        ("selective_risk", "area_under_risk_coverage"),
        "lower_is_better",
    ),
}


def _nested_number(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if current is None or isinstance(current, bool):
        return None
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


def _stable_bootstrap_seed(base_seed: int, *parts: str) -> int:
    digest = sha256_json([base_seed, *parts])
    return int(digest[:16], 16) % (2**32)


def _hierarchical_profile_bootstrap(
    variant_values: Mapping[str, Mapping[int, float]],
    reference_values: Mapping[str, Mapping[int, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Pair profiles first and resample technical seeds only within profile."""

    common_profiles = sorted(set(variant_values) & set(reference_values))
    paired: dict[str, list[float]] = {}
    for profile in common_profiles:
        seeds = sorted(
            set(variant_values[profile]) & set(reference_values[profile])
        )
        differences = [
            float(variant_values[profile][item] - reference_values[profile][item])
            for item in seeds
        ]
        if differences:
            paired[profile] = differences
    profiles = sorted(paired)
    profile_means = [float(np.mean(paired[name])) for name in profiles]
    if not profiles:
        return {
            "status": "not_applicable",
            "reason": "no common profile/seed endpoint values",
            "profiles": 0,
        }
    point = float(np.mean(profile_means))
    if len(profiles) < 2:
        return {
            "status": "insufficient_profiles",
            "paired_mean_difference": point,
            "profiles": len(profiles),
            "profile_mean_differences": dict(zip(profiles, profile_means, strict=True)),
            "bootstrap_95_interval": None,
        }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        sampled_profiles = rng.integers(0, len(profiles), size=len(profiles))
        values: list[float] = []
        for profile_index in sampled_profiles:
            differences = np.asarray(paired[profiles[int(profile_index)]])
            sampled_seeds = rng.integers(0, len(differences), size=len(differences))
            values.append(float(differences[sampled_seeds].mean()))
        bootstrap[iteration] = float(np.mean(values))
    return {
        "status": "reported",
        "paired_mean_difference": point,
        "profiles": len(profiles),
        "technical_seed_counts": {
            name: len(paired[name]) for name in profiles
        },
        "profile_mean_differences": dict(zip(profiles, profile_means, strict=True)),
        "bootstrap_95_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_repetitions": repetitions,
        "resampling": (
            "paired plant_profile_id clusters; technical seeds resampled within profile"
        ),
    }


def _aggregate_results(
    records: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize seeds secondarily and profiles as the inferential unit."""

    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_variant.setdefault(str(record["variant"]["variant_id"]), []).append(record)
    seed_summaries: dict[str, Any] = {}
    profile_values: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for variant_id, variant_records in by_variant.items():
        seed_summaries[variant_id] = {}
        profile_values[variant_id] = {name: {} for name in ENDPOINTS}
        for endpoint, (scope, path, direction) in ENDPOINTS.items():
            values = [
                value
                for record in variant_records
                if (
                    value := _nested_number(
                        record["metrics"]["overall"]
                        if scope == "overall"
                        else record["metrics"]["by_split"].get(scope, {}),
                        path,
                    )
                )
                is not None
            ]
            seed_summaries[variant_id][endpoint] = {
                "direction": direction,
                "scope": scope,
                **_mean_sd(values),
                "unit": "technical training seed (secondary reproducibility only)",
            }
            for record in variant_records:
                seed = int(record["seed"])
                for profile, metrics in record["metrics"]["by_profile"].items():
                    if (
                        scope != "overall"
                        and record["metrics"]["profile_splits"].get(profile)
                        != scope
                    ):
                        continue
                    value = _nested_number(metrics, path)
                    if value is not None:
                        profile_values[variant_id][endpoint].setdefault(
                            profile, {}
                        )[seed] = value

    aggregation = contract["aggregation"]
    repetitions = int(aggregation["bootstrap_repetitions"])
    base_seed = int(aggregation["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    reference = profile_values.get(FULL_MODEL_NAME, {})
    for variant_id, endpoints in profile_values.items():
        if variant_id == FULL_MODEL_NAME:
            continue
        comparisons[variant_id] = {}
        for endpoint, (_scope, _path, direction) in ENDPOINTS.items():
            result = _hierarchical_profile_bootstrap(
                endpoints[endpoint],
                reference.get(endpoint, {}),
                repetitions=repetitions,
                seed=_stable_bootstrap_seed(base_seed, variant_id, endpoint),
            )
            result["difference"] = "variant_minus_flowtwin_guard"
            result["direction"] = direction
            comparisons[variant_id][endpoint] = result
    return {
        "inferential_unit": "plant_profile_id",
        "technical_repeat": "deterministic training seed",
        "seed_reproducibility_summaries": seed_summaries,
        "paired_profile_comparisons": comparisons,
        "ood_note": (
            "OOD metrics remain descriptive per run here; a single profile cannot "
            "supply both ID and OOD labels, so row-level OOD AUROC is not promoted "
            "to a profile-independent inferential endpoint."
        ),
    }


def _tier_assessment(
    variants: Sequence[Variant],
    budget: Mapping[str, Any],
    contract: Mapping[str, Any],
    test_splits: Sequence[str],
    device: torch.device,
    domain_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registered_ids = {item.variant_id for item in registered_variants(contract)}
    selected_ids = {item.variant_id for item in variants}
    registered_budget = dict(contract["training_budget"])
    registered_budget["alpha"] = float(
        contract["calibration"]["conformal_alpha"]
    )
    registered_budget["ood_alpha"] = float(contract["calibration"]["ood_alpha"])
    comparable_fields = (
        "window_size",
        "stride",
        "train_windows_per_episode",
        "batch_size",
        "observer_epochs",
        "diagnostic_epochs",
        "hidden_dim",
        "observer_hidden_dim",
        "layers",
        "attention_heads",
        "dropout",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "alpha",
        "ood_alpha",
    )
    budget_differences = {
        name: {"registered": registered_budget[name], "actual": budget[name]}
        for name in comparable_fields
        if budget[name] != registered_budget[name]
    }
    feature_expected = contract["dataset_contract"]["feature_set"]
    if budget["feature_set"] != feature_expected:
        budget_differences["feature_set"] = {
            "registered": feature_expected,
            "actual": budget["feature_set"],
        }
    registered_seeds = list(map(int, registered_budget["seeds"]))
    actual_seeds = list(map(int, budget["seeds"]))
    all_variants = selected_ids == registered_ids
    exact_nonseed_budget = not budget_differences
    exact_seeds = actual_seeds == registered_seeds
    registered_seed_subset = bool(actual_seeds) and set(actual_seeds) <= set(
        registered_seeds
    )
    cpu_reference = device.type == "cpu"
    has_id = "test_id" in test_splits
    has_ood = "test_ood_profile" in test_splits
    verified_ood = bool(
        has_ood
        and domain_audit
        and domain_audit.get("domain_contract_verified")
        and domain_audit.get("profile_table_domain_verified")
        and domain_audit.get("profile_parameter_ranges_verified")
        and domain_audit.get("observed_ood_domains")
    )
    if (
        all_variants
        and exact_nonseed_budget
        and exact_seeds
        and cpu_reference
        and has_id
        and verified_ood
    ):
        tier = "protocol_complete_synthetic"
    elif (
        all_variants
        and exact_nonseed_budget
        and registered_seed_subset
        and cpu_reference
        and has_id
    ):
        tier = "pilot"
    else:
        tier = "development"
    return {
        "tier": tier,
        "all_registered_variants": all_variants,
        "exact_registered_nonseed_budget": exact_nonseed_budget,
        "exact_registered_seeds": exact_seeds,
        "registered_seed_subset": registered_seed_subset,
        "reference_cpu": cpu_reference,
        "has_test_id": has_id,
        "has_test_ood_profile": has_ood,
        "verified_ood_domain_contract": verified_ood,
        "external_confirmatory_claim": False,
        "external_confirmatory_reason": (
            "requires prospective immutable preregistration, profile-level power, "
            "and a sealed independent/external dataset; this synthetic runner never "
            "grants confirmatory status automatically"
        ),
        "budget_differences": budget_differences,
        "missing_variants": sorted(registered_ids - selected_ids),
        "extra_variants": sorted(selected_ids - registered_ids),
        "publication_status": {
            "development": "pipeline/debug evidence only",
            "pilot": "registered ID-pilot evidence; not confirmatory OOD evidence",
            "protocol_complete_synthetic": (
                "matches the frozen synthetic matrix/budget and contains verified "
                "ID+OOD partitions; not external or field confirmation"
            ),
        }[tier],
    }


def _artifact_paths(directory: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )


def _validate_v03_candidate_bindings(
    candidate_contract: Mapping[str, Any],
    *,
    candidate_contract_path: Path,
    benchmark_contract_path: Path,
    dataset: Path,
    splits: Path,
    cache: Path | None,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an experimental run to the disclosed, already-opened D2 inputs."""

    parent = candidate_contract["base_benchmark_contract"]
    development = candidate_contract["development_dataset"]
    checks = {
        "base_benchmark_contract_sha256": {
            "expected": str(parent["sha256"]),
            "actual": sha256_file(benchmark_contract_path),
        },
        "dataset_manifest_sha256": {
            "expected": str(development["dataset_manifest_sha256"]),
            "actual": sha256_file(Path(dataset) / "dataset_manifest.json"),
        },
        "split_manifest_sha256": {
            "expected": str(development["split_manifest_sha256"]),
            "actual": sha256_file(Path(splits) / "split_manifest.json"),
        },
    }
    if cache is None:
        raise ValueError("v0.3 candidate contract requires the bound D2 cache")
    checks["cache_manifest_sha256"] = {
        "expected": str(development["cache_manifest_sha256"]),
        "actual": sha256_file(Path(cache) / "cache_manifest.json"),
    }
    mismatches = [
        name
        for name, values in checks.items()
        if values["expected"] != values["actual"]
    ]
    if mismatches:
        raise ValueError(f"v0.3 candidate artifact binding mismatch: {mismatches}")
    expected_budget = dict(candidate_contract["training_budget"])
    if dict(budget) != expected_budget:
        changed = sorted(
            name
            for name in set(budget) | set(expected_budget)
            if budget.get(name) != expected_budget.get(name)
        )
        raise ValueError(
            "v0.3 candidate fixed budget cannot be overridden; changed fields: "
            f"{changed}"
        )
    manifest = json.loads(
        (Path(dataset) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("dataset_version") != development["dataset_version"]:
        raise ValueError("v0.3 candidate dataset_version binding mismatch")
    return {
        "status": "pass",
        "candidate_contract_path": str(Path(candidate_contract_path).resolve()),
        "candidate_contract_sha256": sha256_file(candidate_contract_path),
        "checks": checks,
        "budget_exact": True,
        "d2_test_seen_before_freeze": True,
        "evidence_status": "post_hoc_development_only",
        "eligible_for_confirmatory_claims": False,
    }


def run(args: argparse.Namespace) -> Path:
    contract = load_benchmark_contract(args.contract)
    variants = resolve_variants(args.variants, contract)
    candidate_selected = any(
        variant.variant_id == V03_CANDIDATE_NAME for variant in variants
    )
    requested_candidate_contract = getattr(args, "candidate_contract", None)
    candidate_contract_path: Path | None = None
    candidate_contract: Mapping[str, Any] | None = None
    if candidate_selected or requested_candidate_contract is not None:
        candidate_contract_path = Path(
            requested_candidate_contract or DEFAULT_V03_CANDIDATE_CONTRACT
        )
        candidate_contract = load_v03_candidate_contract(candidate_contract_path)
        selected_ids = {variant.variant_id for variant in variants}
        planned_ids = set(
            map(str, candidate_contract["planned_comparison_variants"])
        )
        if not selected_ids <= planned_ids:
            raise ValueError(
                "v0.3 candidate contract only permits its planned candidate/"
                "comparator rows"
            )
        registered_ids = {
            variant.variant_id for variant in registered_variants(contract)
        }
        if V03_CANDIDATE_NAME in registered_ids:
            raise ValueError("v0.3 candidate must not enter the frozen v0.2 registry")
    budget = _effective_budget(args, contract, candidate_contract)
    _validate_budget(budget)
    for directory in (args.dataset, args.splits):
        failures = verify_checksums(directory)
        if failures:
            raise ValueError(f"checksum verification failed for {directory}: {failures}")
    domain_audit = _validate_split_domain_contract(args.dataset, args.splits)
    candidate_binding_audit: Mapping[str, Any] | None = None
    if candidate_contract is not None:
        assert candidate_contract_path is not None
        candidate_binding_audit = _validate_v03_candidate_bindings(
            candidate_contract,
            candidate_contract_path=candidate_contract_path,
            benchmark_contract_path=Path(args.contract),
            dataset=Path(args.dataset),
            splits=Path(args.splits),
            cache=Path(args.cache) if args.cache is not None else None,
            budget=budget,
        )
    split_names = sorted(
        {row["split"] for row in read_csv_rows(args.splits / "episode_splits.csv")}
    )
    for required in ("train_id", "validation_id"):
        if required not in split_names:
            raise ValueError(f"split dataset lacks required partition {required}")
    test_splits = tuple(name for name in split_names if name.startswith("test_"))
    if not test_splits:
        raise ValueError("split dataset has no test_* partition")
    device = _resolve_device(args.device)
    graph = build_process_graph(feature_set=str(budget["feature_set"]))
    targets = load_target_contract(graph=graph)
    cache_manifest: Mapping[str, Any] | None = None
    if args.cache is not None:
        cache_manifest = load_cache_manifest(
            args.cache, args.dataset, args.splits, graph, targets
        )
        standardizer = FeatureStandardizer.from_dict(cache_manifest["standardizer"])
    else:
        standardizer = FeatureStandardizer.fit(args.dataset, args.splits, graph)
    context = RuntimeContext(
        dataset=Path(args.dataset),
        splits=Path(args.splits),
        cache=Path(args.cache) if args.cache is not None else None,
        cache_manifest=cache_manifest,
        graph=graph,
        targets=targets,
        standardizer=standardizer,
        budget=budget,
        device=device,
        candidate_contract=candidate_contract,
    )
    tier = _tier_assessment(
        variants,
        budget,
        contract,
        test_splits,
        device,
        domain_audit,
    )
    if candidate_contract is not None and tier["tier"] != "development":
        raise ValueError("a v0.3 candidate-contract run must always remain development")
    temporary, existed_empty = _prepare_output(args.output)
    try:
        write_json(temporary / "benchmark_contract.json", contract)
        if candidate_contract is not None:
            write_json(temporary / "candidate_contract.json", candidate_contract)
            assert candidate_binding_audit is not None
            write_json(
                temporary / "candidate_binding_audit.json",
                candidate_binding_audit,
            )
        write_json(temporary / "standardizer.json", standardizer.to_dict())
        write_json(temporary / "graph_contract.json", graph.to_dict())
        write_json(temporary / "split_domain_audit.json", domain_audit)
        training_coverage = _training_coverage_audit(context)
        write_json(temporary / "training_coverage_audit.json", training_coverage)
        critical_delay_coverage: Mapping[str, Any] | None = None
        if candidate_contract is not None:
            critical_delay_coverage = _critical_delay_coverage_audit(
                context, candidate_contract
            )
            write_json(
                temporary / "critical_delay_coverage_audit.json",
                critical_delay_coverage,
            )
            if critical_delay_coverage["status"] != "pass":
                raise ValueError(
                    "v0.3 critical-delay coverage gate failed before training"
                )
        target_weights, target_weight_report = _target_weights(context)
        write_json(temporary / "target_weights.json", target_weight_report)
        effective = {
            "benchmark_runner_version": BENCHMARK_RUNNER_VERSION,
            "benchmark_contract_sha256": sha256_file(args.contract),
            "candidate_contract": (
                None
                if candidate_contract_path is None
                else {
                    "path": str(candidate_contract_path.resolve()),
                    "sha256": sha256_file(candidate_contract_path),
                    "status": "post_hoc_development_only",
                    "d2_test_seen_before_freeze": True,
                }
            ),
            "variants": [variant.to_dict() for variant in variants],
            "budget": budget,
            "device": str(device),
            "cache": str(Path(args.cache).resolve()) if args.cache else None,
            "save_predictions": bool(args.save_predictions),
            "test_splits": list(test_splits),
            "tier_assessment": tier,
            "critical_delay_coverage_audit": critical_delay_coverage,
            "train_validation_test_firewall": (
                "train fixed-budget weights -> validation-only calibration -> "
                "single exhaustive test evaluation"
            ),
        }
        write_json(temporary / "benchmark_config.json", effective)
        records: list[dict[str, Any]] = []
        for variant in variants:
            for seed in budget["seeds"]:
                records.append(
                    _run_variant(
                        variant,
                        int(seed),
                        context,
                        target_weights,
                        contract,
                        temporary,
                        save_predictions=bool(args.save_predictions),
                    )
                )
        results = {
            "benchmark_runner_version": BENCHMARK_RUNNER_VERSION,
            "benchmark_id": contract["benchmark_id"],
            "tier_assessment": tier,
            "training_coverage_audit": training_coverage,
            "split_domain_audit": domain_audit,
            "candidate_binding_audit": candidate_binding_audit,
            "critical_delay_coverage_audit": critical_delay_coverage,
            "runs": records,
            "aggregation": _aggregate_results(records, contract),
            "claim_boundary": contract["claim_boundary"],
        }
        if candidate_contract is not None:
            results["candidate_claim_boundary"] = candidate_contract[
                "claim_boundary"
            ]
            results["future_confirmatory_gate"] = candidate_contract[
                "future_confirmatory_gate"
            ]
        write_json(temporary / "benchmark_results.json", results)
        source_names = (
            "run_flowtwin_benchmark.py",
            "ml_pipeline_common.py",
            "reference_plant.py",
            "sensor_calibration.py",
            "fault_asset_contract.json",
            "ml_contract.json",
            "reference_pid.json",
            "sensor_catalog.json",
            "flowtwin_guard/graph.py",
            "flowtwin_guard/data.py",
            "flowtwin_guard/cache.py",
            "flowtwin_guard/model.py",
            "flowtwin_guard/hybrid.py",
            "flowtwin_guard/alarm.py",
            "flowtwin_guard/baselines.py",
            "flowtwin_guard/ablation.py",
            "flowtwin_guard/dspr.py",
            "flowtwin_guard/conformal.py",
            "flowtwin_guard/metrics.py",
        )
        manifest = {
            "benchmark_runner_version": BENCHMARK_RUNNER_VERSION,
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "model_version": MODEL_VERSION,
            "hybrid_model_version": HYBRID_MODEL_VERSION,
            "alarm_policy_version": ALARM_POLICY_VERSION,
            "baseline_version": BASELINE_VERSION,
            "ablation_version": ABLATION_VERSION,
            "metrics_version": METRICS_VERSION,
            "tier_assessment": tier,
            "benchmark_contract_source": {
                "path": str(Path(args.contract).resolve()),
                "sha256": sha256_file(args.contract),
            },
            "candidate_contract_source": (
                None
                if candidate_contract_path is None
                else {
                    "path": str(candidate_contract_path.resolve()),
                    "sha256": sha256_file(candidate_contract_path),
                    "version": V03_CANDIDATE_CONTRACT_VERSION,
                    "d2_test_seen_before_freeze": True,
                    "eligible_for_confirmatory_claims": False,
                }
            ),
            "dataset_manifest_sha256": sha256_file(
                args.dataset / "dataset_manifest.json"
            ),
            "split_manifest_sha256": sha256_file(
                args.splits / "split_manifest.json"
            ),
            "dataset_checksums_sha256": sha256_file(
                args.dataset / "checksums.sha256"
            ),
            "split_checksums_sha256": sha256_file(
                args.splits / "checksums.sha256"
            ),
            "cache_manifest_sha256": (
                sha256_file(args.cache / "cache_manifest.json") if args.cache else None
            ),
            "input_contract": {
                "feature_set": budget["feature_set"],
                "features": list(graph.features),
                "route_controls": list(CONTROL_FIELDS),
                "forbidden_feature_intersection": [],
                "oracle_fields": "target/evaluation only",
            },
            "split_policy": {
                "training": ["train_id"],
                "calibration": ["validation_id"],
                "evaluation": list(test_splits),
                "test_access_after_calibration": True,
            },
            "statistical_unit": {
                "inferential": "plant_profile_id",
                "technical_repeat": "training seed",
                "time_rows_are_independent_replicates": False,
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "provenance": {
                **{name: sha256_file(BASE_DIR / name) for name in source_names},
                "benchmark_contract": sha256_file(args.contract),
                **(
                    {}
                    if candidate_contract_path is None
                    else {
                        "candidate_contract": sha256_file(
                            candidate_contract_path
                        )
                    }
                ),
            },
            "claim_boundary": contract["claim_boundary"],
        }
        if candidate_contract is not None:
            manifest["candidate_binding_audit"] = candidate_binding_audit
            manifest["critical_delay_coverage_audit"] = critical_delay_coverage
            manifest["candidate_claim_boundary"] = candidate_contract[
                "claim_boundary"
            ]
        write_json(temporary / "run_manifest.json", manifest)
        write_checksums(temporary, _artifact_paths(temporary))
        _commit_output(temporary, Path(args.output), existed_empty)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Path(args.output).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    results = json.loads((output / "benchmark_results.json").read_text(encoding="utf-8"))
    print(
        f"FlowTwin benchmark complete: {output}\n"
        f"tier={results['tier_assessment']['tier']}, "
        f"runs={len(results['runs'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
