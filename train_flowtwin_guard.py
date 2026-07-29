#!/usr/bin/env python3
"""Train, calibrate and evaluate FlowTwin-Guard on an HTST D1 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised without ML extras
    raise SystemExit(
        "FlowTwin-Guard requires the optional ML environment. Run: "
        "python3.12 -m venv .venv && "
        ".venv/bin/pip install -r requirements-ml.txt"
    ) from exc

from flowtwin_guard.conformal import (
    ModeConformalCalibrator,
    counterfactual_group_mode_block_max,
)
from flowtwin_guard.cache import load_cache_manifest
from flowtwin_guard.data import (
    FeatureStandardizer,
    HTSTWindowDataset,
    TargetContract,
    load_target_contract,
    normalized_sensor_uncertainty,
)
from flowtwin_guard.graph import CONTROL_FIELDS, ProcessGraph, build_process_graph
from flowtwin_guard.model import (
    MODEL_VERSION,
    FlowTwinConfig,
    FlowTwinGuard,
    counterfactual_contrastive_loss,
    focal_cross_entropy,
)
from ml_pipeline_common import (
    BASE_DIR,
    read_csv_rows,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_csv_header,
    write_json,
)


TRAINER_VERSION = "0.4.0"
ARTIFACTS = (
    "training_config.json",
    "training_history.json",
    "graph_contract.json",
    "standardizer.json",
    "model.pt",
    "conformal_state.json",
    "predictions.csv",
    "metrics.json",
    "run_manifest.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Optional checksummed per-episode cache created by build_flowtwin_cache.py",
    )
    parser.add_argument("--feature-set", default="S3-context")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--observer-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--observer-hidden-dim", type=int, default=48)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--ood-alpha", type=float, default=0.01)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument(
        "--train-windows-per-episode",
        type=int,
        default=0,
        help="Deterministic timeline-spanning cap applied separately to every train episode",
    )
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    integer_positive = (
        "window_size",
        "stride",
        "batch_size",
        "observer_epochs",
        "epochs",
        "hidden_dim",
        "observer_hidden_dim",
        "graph_layers",
    )
    for name in integer_positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stride > args.window_size:
        raise ValueError("--stride cannot exceed --window-size")
    if (
        args.max_train_windows < 0
        or args.max_eval_windows < 0
        or args.train_windows_per_episode < 0
    ):
        raise ValueError("maximum window counts must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1)")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    for name in ("alpha", "ood_alpha"):
        if not 0.0 < float(getattr(args, name)) < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0,1)")


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
    path = path.resolve()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    existed_empty = False
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
        existed_empty = True
    temporary = Path(tempfile.mkdtemp(prefix=".flowtwin-", dir=parent))
    return temporary, existed_empty


def _commit_output(temporary: Path, output: Path, existed_empty: bool) -> None:
    output = output.resolve()
    if existed_empty:
        output.rmdir()
    os.replace(temporary, output)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _dataset(
    args: argparse.Namespace,
    graph: ProcessGraph,
    targets: TargetContract,
    standardizer: FeatureStandardizer,
    splits: Sequence[str],
    *,
    normal_only: bool = False,
    evaluation: bool = False,
) -> HTSTWindowDataset:
    return HTSTWindowDataset(
        args.dataset,
        args.splits,
        graph,
        targets,
        standardizer,
        accepted_splits=splits,
        window_size=args.window_size,
        stride=args.stride,
        normal_only=normal_only,
        max_windows=(args.max_eval_windows if evaluation else args.max_train_windows),
        cache=args.cache,
        cache_manifest=getattr(args, "_cache_manifest", None),
        max_windows_per_episode=(
            0 if evaluation else args.train_windows_per_episode
        ),
    )


def _loader(dataset: HTSTWindowDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )


def _observer_pretrain(
    model: FlowTwinGuard,
    args: argparse.Namespace,
    graph: ProcessGraph,
    targets: TargetContract,
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> list[dict[str, float]]:
    model.set_observer_trainable(True)
    optimizer = torch.optim.AdamW(
        model.observer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(args.observer_epochs):
        model.observer.train()
        total = 0.0
        batches = 0
        dataset = _dataset(
            args,
            graph,
            targets,
            standardizer,
            ("train_id",),
            normal_only=True,
        )
        for raw_batch in _loader(dataset, args):
            batch = _move_batch(raw_batch, device)
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
            torch.nn.utils.clip_grad_norm_(model.observer.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("nominal observer received no training windows")
        history.append({"epoch": epoch + 1, "loss": total / batches})
    model.set_observer_trainable(False)
    model.observer.eval()
    return history


def _transport_loss(
    transport_error: Tensor,
    class_target: Tensor,
    valid_mask: Tensor,
    normal_index: int,
) -> Tensor:
    selected = valid_mask & (class_target == normal_index)
    if not torch.any(selected):
        return transport_error.sum() * 0.0
    return transport_error[selected].mean()


def _balanced_target_weights(
    args: argparse.Namespace,
    graph: ProcessGraph,
    targets: TargetContract,
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any], set[str]]:
    """Compute inverse-square-root weights from train rows, counted once."""

    widths = {
        "class": len(targets.codes),
        "anomaly": 2,
        "location": len(targets.locations),
        "mechanism": len(targets.mechanisms),
    }
    fields = {
        "class": "class_target",
        "anomaly": "anomaly_target",
        "location": "location_target",
        "mechanism": "mechanism_target",
    }
    counts = {name: np.zeros(width, dtype=np.int64) for name, width in widths.items()}
    target_episode_ids: set[str] = set()
    dataset = HTSTWindowDataset(
        args.dataset,
        args.splits,
        graph,
        targets,
        standardizer,
        accepted_splits=("train_id",),
        window_size=args.window_size,
        stride=args.stride,
        max_windows=0,
        cache=args.cache,
        cache_manifest=getattr(args, "_cache_manifest", None),
    )
    for batch in _loader(dataset, args):
        selected = batch["valid_mask"] & batch["eval_mask"]
        for batch_index, episode_id in enumerate(batch["episode_id"]):
            episode_selected = selected[batch_index]
            if torch.any(
                batch["class_target"][batch_index][episode_selected]
                != targets.normal_index
            ):
                target_episode_ids.add(str(episode_id))
        for name, field in fields.items():
            values = batch[field][selected].numpy()
            counts[name] += np.bincount(values, minlength=widths[name])
    if counts["class"].sum() == 0:
        raise ValueError("cannot derive target weights from an empty training split")
    weights: dict[str, Tensor] = {}
    report: dict[str, Any] = {}
    for name, values in counts.items():
        observed = values > 0
        raw = np.zeros(len(values), dtype=np.float64)
        raw[observed] = np.sqrt(values[observed].sum() / (observed.sum() * values[observed]))
        if observed.any():
            raw[observed] /= raw[observed].mean()
        raw = np.clip(raw, 0.25, 4.0)
        raw[~observed] = 0.0
        weights[name] = torch.tensor(raw, dtype=torch.float32, device=device)
        report[name] = {"counts": values.tolist(), "weights": raw.tolist()}
    return weights, report, target_episode_ids


def _training_sampling_audit(
    args: argparse.Namespace,
    graph: ProcessGraph,
    targets: TargetContract,
    standardizer: FeatureStandardizer,
    target_episode_ids: set[str],
) -> dict[str, Any]:
    """Fail if a capped train iterator erases any target-bearing episode."""

    expected_train = {
        row["episode_id"]
        for row in read_csv_rows(args.splits / "episode_splits.csv")
        if row["split"] == "train_id"
    }
    observed_train: set[str] = set()
    sampled_target: set[str] = set()
    sampled_windows = 0
    dataset = _dataset(args, graph, targets, standardizer, ("train_id",))
    for batch in _loader(dataset, args):
        sampled_windows += len(batch["episode_id"])
        selected = batch["valid_mask"] & batch["eval_mask"]
        for batch_index, episode_id in enumerate(batch["episode_id"]):
            episode_id = str(episode_id)
            observed_train.add(episode_id)
            if torch.any(
                batch["class_target"][batch_index][selected[batch_index]]
                != targets.normal_index
            ):
                sampled_target.add(episode_id)
    missing_train = sorted(expected_train - observed_train)
    missing_target = sorted(target_episode_ids - sampled_target)
    if missing_train:
        raise ValueError(
            f"training sampler omitted {len(missing_train)} train episodes"
        )
    if missing_target:
        raise ValueError(
            "training sampler omitted non-normal target rows for "
            f"{len(missing_target)} episodes: {missing_target[:5]}"
        )
    return {
        "policy": "per-episode target onset/middle/end priority, then deterministic farthest-point timeline coverage",
        "target_source": "train_id class targets only; no validation/test labels",
        "optimizer_context_policy": (
            "overlap is encoder context only; loss uses valid_mask & eval_mask"
        ),
        "train_episodes": len(expected_train),
        "sampled_windows": sampled_windows,
        "target_bearing_episodes": len(target_episode_ids),
        "target_bearing_episodes_covered": len(sampled_target & target_episode_ids),
        "missing_train_episodes": 0,
        "missing_target_bearing_episodes": 0,
    }


def _diagnostic_train(
    model: FlowTwinGuard,
    args: argparse.Namespace,
    graph: ProcessGraph,
    targets: TargetContract,
    standardizer: FeatureStandardizer,
    device: torch.device,
    target_weights: Mapping[str, Tensor],
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        model.observer.eval()
        totals: Counter[str] = Counter()
        batches = 0
        dataset = _dataset(args, graph, targets, standardizer, ("train_id",))
        for raw_batch in _loader(dataset, args):
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["x"],
                batch["controls_raw"],
                batch["dt_s"],
                batch["edge_volume_l"],
                batch["valid_mask"],
            )
            reference = model(
                batch["reference_x"],
                batch["reference_controls_raw"],
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
                "counterfactual": counterfactual_contrastive_loss(
                    output["embedding"],
                    reference["embedding"],
                    batch["class_target"],
                    valid,
                    normal_index=targets.normal_index,
                ),
                "transport": _transport_loss(
                    output["transport_error"],
                    batch["class_target"],
                    valid,
                    targets.normal_index,
                ),
                "delay_regularization": model.delay_regularization(),
            }
            loss = (
                losses["class"]
                + 0.50 * losses["anomaly"]
                + 0.25 * losses["location"]
                + 0.25 * losses["mechanism"]
                + 0.10 * losses["counterfactual"]
                + 0.01 * losses["transport"]
                + 0.001 * losses["delay_regularization"]
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite diagnostic training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            totals["total"] += float(loss.detach().cpu())
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("diagnostic model received no training windows")
        history.append(
            {"epoch": epoch + 1, **{name: value / batches for name, value in totals.items()}}
        )
    return history


def _save_checkpoint(
    path: Path,
    model: FlowTwinGuard,
    config: FlowTwinConfig,
    graph: ProcessGraph,
    targets: TargetContract,
) -> None:
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "config": config.to_dict(),
            "graph_version": graph.version,
            "plant_id": graph.plant_id,
            "features": list(graph.features),
            "codes": list(targets.codes),
            "locations": list(targets.locations),
            "mechanisms": list(targets.mechanisms),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _reload_checkpoint(
    path: Path,
    graph: ProcessGraph,
    targets: TargetContract,
    uncertainty: Tensor,
    device: torch.device,
) -> FlowTwinGuard:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint["model_version"] != MODEL_VERSION
        or tuple(checkpoint["features"]) != graph.features
        or tuple(checkpoint["codes"]) != targets.codes
    ):
        raise ValueError("checkpoint contract does not match runtime graph/targets")
    config = FlowTwinConfig.from_dict(checkpoint["config"])
    model = FlowTwinGuard(graph, config, uncertainty.to(device)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.set_observer_trainable(False)
    model.eval()
    return model


def _collect(
    model: FlowTwinGuard,
    dataset: HTSTWindowDataset,
    args: argparse.Namespace,
    targets: TargetContract,
    device: torch.device,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    anomaly_probability: list[float] = []
    location_prediction: list[int] = []
    mechanism_prediction: list[int] = []
    class_targets: list[int] = []
    modes: list[float] = []
    energies: list[float] = []
    splits: list[str] = []
    counterfactual_groups: list[str] = []
    plant_profiles: list[str] = []
    rows: list[dict[str, Any]] = []
    cip_index = CONTROL_FIELDS.index("cip_cycle_active")
    model.eval()
    with torch.no_grad():
        for raw_batch in _loader(dataset, args):
            batch = _move_batch(raw_batch, device)
            output = model(
                batch["x"],
                batch["controls_raw"],
                batch["dt_s"],
                batch["edge_volume_l"],
                batch["valid_mask"],
            )
            class_probability = torch.softmax(output["class_logits"], dim=-1)
            anomaly_prob = torch.softmax(output["anomaly_logits"], dim=-1)[..., 1]
            location = output["location_logits"].argmax(dim=-1)
            mechanism = output["mechanism_logits"].argmax(dim=-1)
            selected = (batch["valid_mask"] & batch["eval_mask"]).detach().cpu()
            batch_size = selected.shape[0]
            for batch_index in range(batch_size):
                positions = torch.nonzero(selected[batch_index], as_tuple=False).flatten()
                for position_tensor in positions:
                    position = int(position_tensor)
                    probability = class_probability[batch_index, position].detach().cpu().numpy()
                    target_index = int(batch["class_target"][batch_index, position].cpu())
                    mode = float(batch["controls_raw"][batch_index, position, cip_index].cpu())
                    energy = float(output["ood_energy"][batch_index, position].cpu())
                    probabilities.append(probability)
                    anomaly_probability.append(
                        float(anomaly_prob[batch_index, position].cpu())
                    )
                    location_prediction.append(
                        int(location[batch_index, position].cpu())
                    )
                    mechanism_prediction.append(
                        int(mechanism[batch_index, position].cpu())
                    )
                    class_targets.append(target_index)
                    modes.append(mode)
                    energies.append(energy)
                    split = str(raw_batch["split"][batch_index])
                    splits.append(split)
                    counterfactual_groups.append(
                        str(raw_batch["counterfactual_group_id"][batch_index])
                    )
                    plant_profiles.append(
                        str(raw_batch["plant_profile_id"][batch_index])
                    )
                    rows.append(
                        {
                            "episode_id": str(raw_batch["episode_id"][batch_index]),
                            "split": split,
                            "time_s": float(batch["time_s"][batch_index, position].cpu()),
                            "step_dt_s": float(batch["dt_s"][batch_index, position].cpu()),
                            "episode_code": str(raw_batch["canonical_code"][batch_index]),
                            "target_code": targets.codes[target_index],
                            "fault_active": int(
                                batch["fault_active"][batch_index, position].cpu() >= 0.5
                            ),
                            "safety_event": int(
                                batch["safety_event"][batch_index, position].cpu() >= 0.5
                            ),
                            "unsafe_forward_l": float(
                                batch["unsafe_forward_l"][batch_index, position].cpu()
                            ),
                            "detection_eligible": int(
                                raw_batch["detection_eligible"][batch_index]
                            ),
                            "effect_time_s": float(raw_batch["effect_time_s"][batch_index]),
                        }
                    )
    if not probabilities:
        raise ValueError("evaluation produced no rows")
    return {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "anomaly_probability": np.asarray(anomaly_probability, dtype=np.float64),
        "location_prediction": np.asarray(location_prediction, dtype=np.int64),
        "mechanism_prediction": np.asarray(mechanism_prediction, dtype=np.int64),
        "targets": np.asarray(class_targets, dtype=np.int64),
        "modes": np.asarray(modes, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "splits": splits,
        "counterfactual_groups": counterfactual_groups,
        "plant_profiles": plant_profiles,
        "rows": rows,
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    log_probability = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probability -= log_probability.max(axis=1, keepdims=True)
    scaled = np.exp(log_probability)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Fit one deterministic validation-only temperature by NLL grid search."""

    candidates = np.exp(np.linspace(math.log(0.05), math.log(5.0), 241))
    best_temperature = 1.0
    best_nll = math.inf
    indexes = np.arange(len(targets))
    for candidate in candidates:
        scaled = _temperature_scale(probabilities, float(candidate))
        nll = float(-np.log(np.clip(scaled[indexes, targets], 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(candidate)
    return best_temperature


def _fit_anomaly_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    """Select a validation-only F1 threshold with a conservative tie break."""

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = _binary_metrics(truth, probability >= threshold)
        f1 = float(metrics["f1"] or 0.0)
        if f1 > best_f1 + 1e-12 or (
            abs(f1 - best_f1) <= 1e-12 and threshold > best_threshold
        ):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def _classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, class_count: int
) -> dict[str, Any]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, predicted in zip(truth, prediction, strict=True):
        confusion[int(target), int(predicted)] += 1
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for index in range(class_count):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if support:
            precision_for_f1 = precision if precision is not None else 0.0
            recall_for_f1 = recall if recall is not None else 0.0
            f1 = (
                2.0
                * precision_for_f1
                * recall_for_f1
                / (precision_for_f1 + recall_for_f1)
                if precision_for_f1 + recall_for_f1 > 0.0
                else 0.0
            )
            f1_values.append(f1)
        else:
            f1 = None
        per_class.append(
            {
                "class_index": index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "rows": int(truth.size),
        "accuracy": float((truth == prediction).mean()),
        "macro_f1_observed_classes": (
            float(sum(f1_values) / len(f1_values)) if f1_values else None
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _event_metrics(
    rows: Sequence[Mapping[str, Any]], alarm: np.ndarray
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Mapping[str, Any], bool]]] = defaultdict(list)
    for row, active in zip(rows, alarm, strict=True):
        grouped[str(row["episode_id"])].append((row, bool(active)))
    target_events = detected = 0
    latencies: list[float] = []
    unsafe_before_detection = 0.0
    false_alarm_steps = 0
    negative_seconds = 0.0
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda item: float(item[0]["time_s"]))
        first = episode_rows[0][0]
        effect = float(first["effect_time_s"])
        eligible = bool(first["detection_eligible"]) and str(first["episode_code"]).startswith("F")
        first_alarm: float | None = None
        if eligible and math.isfinite(effect):
            target_events += 1
            for row, active in episode_rows:
                time_s = float(row["time_s"])
                if active and time_s + 1e-9 >= effect and int(row["fault_active"]):
                    first_alarm = time_s
                    detected += 1
                    latencies.append(max(0.0, time_s - effect))
                    break
            cutoff = first_alarm if first_alarm is not None else math.inf
            unsafe_before_detection += sum(
                float(row["unsafe_forward_l"])
                for row, _active in episode_rows
                if effect <= float(row["time_s"]) < cutoff
            )
        for row, active in episode_rows:
            if not int(row["fault_active"]):
                negative_seconds += float(row["step_dt_s"])
                false_alarm_steps += int(active)
    return {
        "target_events": target_events,
        "detected_events": detected,
        "event_recall": _safe_div(detected, target_events),
        "mean_detection_latency_s": (
            float(sum(latencies) / len(latencies)) if latencies else None
        ),
        "median_detection_latency_s": (
            float(np.median(latencies)) if latencies else None
        ),
        "false_alarm_steps": false_alarm_steps,
        "false_alarm_steps_per_negative_hour": _safe_div(
            false_alarm_steps, negative_seconds / 3600.0
        ),
        "unsafe_forward_l_before_detection": unsafe_before_detection,
    }


def _evaluate_and_write(
    collected: Mapping[str, Any],
    calibrator: ModeConformalCalibrator,
    targets: TargetContract,
    output: Path,
    anomaly_threshold: float,
) -> dict[str, Any]:
    probabilities = collected["probabilities"]
    truth = collected["targets"]
    predicted = probabilities.argmax(axis=1)
    anomaly_probability = collected["anomaly_probability"]
    anomaly_alarm = anomaly_probability >= anomaly_threshold
    truth_anomaly = np.asarray(
        [targets.code_is_fault[int(index)] for index in truth], dtype=np.int64
    )
    coverage = 0
    set_sizes: list[int] = []
    decisions: Counter[str] = Counter()
    prediction_sets: list[tuple[int, ...]] = []
    for probability, mode, energy, target in zip(
        probabilities,
        collected["modes"],
        collected["energies"],
        truth,
        strict=True,
    ):
        result = calibrator.predict(probability, mode, energy)
        prediction_sets.append(result.prediction_set)
        decisions[result.decision] += 1
        set_sizes.append(len(result.prediction_set))
        coverage += int(int(target) in result.prediction_set)

    fields = [
        "episode_id",
        "split",
        "time_s",
        "step_dt_s",
        "episode_code",
        "target_code",
        "predicted_code",
        "predicted_location",
        "predicted_mechanism",
        "anomaly_probability",
        "prediction_set",
        "decision",
        "ood_energy",
        "fault_active",
        "safety_event",
        "unsafe_forward_l",
        "detection_eligible",
        "effect_time_s",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = write_csv_header(handle, fields)
        for index, base in enumerate(collected["rows"]):
            writer.writerow(
                {
                    **base,
                    "predicted_code": targets.codes[int(predicted[index])],
                    "predicted_location": targets.locations[
                        int(collected["location_prediction"][index])
                    ],
                    "predicted_mechanism": targets.mechanisms[
                        int(collected["mechanism_prediction"][index])
                    ],
                    "anomaly_probability": float(anomaly_probability[index]),
                    "prediction_set": "|".join(
                        targets.codes[item] for item in prediction_sets[index]
                    ),
                    "decision": (
                        "UNKNOWN"
                        if float(collected["energies"][index])
                        > float(calibrator.energy_threshold)
                        or len(prediction_sets[index]) == 0
                        else "DIAGNOSE"
                        if len(prediction_sets[index]) == 1
                        else "REVIEW"
                    ),
                    "ood_energy": float(collected["energies"][index]),
                }
            )
    class_metrics = _classification_metrics(truth, predicted, len(targets.codes))
    for item, code in zip(class_metrics["per_class"], targets.codes, strict=True):
        item["code"] = code
    return {
        "evaluation_scope": sorted(set(collected["splits"])),
        "class_diagnosis": class_metrics,
        "anomaly_detection": {
            "threshold": anomaly_threshold,
            "threshold_fit_split": "validation_id",
            **_binary_metrics(truth_anomaly, anomaly_alarm.astype(np.int64)),
        },
        "conformal": {
            "empirical_coverage": coverage / len(truth),
            "mean_prediction_set_size": float(np.mean(set_sizes)),
            "decisions": dict(sorted(decisions.items())),
        },
        "event_detection": _event_metrics(collected["rows"], anomaly_alarm),
        "claim_boundary": (
            "Synthetic pre-validation result. DIAGNOSE/REVIEW/UNKNOWN never authorizes "
            "safe product release or overrides the independent PLC/HACCP governor."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    for directory in (args.dataset, args.splits):
        failures = verify_checksums(directory)
        if failures:
            raise ValueError(f"checksum verification failed for {directory}: {failures}")
    split_names = {row["split"] for row in read_csv_rows(args.splits / "episode_splits.csv")}
    required = {"train_id", "validation_id"}
    if not required <= split_names:
        raise ValueError(f"split dataset lacks {sorted(required - split_names)}")
    test_splits = tuple(sorted(name for name in split_names if name.startswith("test_")))
    if not test_splits:
        raise ValueError("split dataset has no test partition")
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    graph = build_process_graph(feature_set=args.feature_set)
    targets = load_target_contract(graph=graph)
    cache_manifest: Mapping[str, Any] | None = None
    if args.cache is not None:
        cache_manifest = load_cache_manifest(
            args.cache, args.dataset, args.splits, graph, targets
        )
        standardizer = FeatureStandardizer.from_dict(cache_manifest["standardizer"])
        # Dataset instances are recreated every epoch. Reuse the manifest that
        # was fully checksum-verified above instead of rehashing the cache.
        setattr(args, "_cache_manifest", cache_manifest)
    else:
        standardizer = FeatureStandardizer.fit(args.dataset, args.splits, graph)
    uncertainty = torch.tensor(normalized_sensor_uncertainty(graph, standardizer))
    config = FlowTwinConfig(
        feature_count=len(graph.features),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        class_count=len(targets.codes),
        location_count=len(targets.locations),
        mechanism_count=len(targets.mechanisms),
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        observer_hidden_dim=args.observer_hidden_dim,
    )
    target_weights, target_weight_report, target_episode_ids = _balanced_target_weights(
        args, graph, targets, standardizer, device
    )
    sampling_report = _training_sampling_audit(
        args, graph, targets, standardizer, target_episode_ids
    )
    temporary, existed_empty = _prepare_output(args.output)
    try:
        write_json(
            temporary / "training_config.json",
            {
                "trainer_version": TRAINER_VERSION,
                "seed": args.seed,
                "device": str(device),
                "feature_set": args.feature_set,
                "window_size": args.window_size,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "observer_epochs": args.observer_epochs,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "alpha": args.alpha,
                "ood_alpha": args.ood_alpha,
                "max_train_windows": args.max_train_windows,
                "train_windows_per_episode": args.train_windows_per_episode,
                "max_eval_windows": args.max_eval_windows,
                "cache": str(args.cache.resolve()) if args.cache is not None else None,
                "model": config.to_dict(),
                "loss_weights": {
                    "class": 1.0,
                    "anomaly": 0.5,
                    "location": 0.25,
                    "mechanism": 0.25,
                    "counterfactual": 0.1,
                    "transport": 0.01,
                    "delay_regularization": 0.001,
                },
                "class_balance": {
                    "method": "train-only inverse-square-root frequency, mean-normalized, clipped [0.25,4]",
                    **target_weight_report,
                },
                "training_sampling_audit": sampling_report,
            },
        )
        write_json(temporary / "graph_contract.json", graph.to_dict())
        write_json(temporary / "standardizer.json", standardizer.to_dict())
        model = FlowTwinGuard(graph, config, uncertainty).to(device)
        observer_history = _observer_pretrain(
            model, args, graph, targets, standardizer, device
        )
        diagnostic_history = _diagnostic_train(
            model,
            args,
            graph,
            targets,
            standardizer,
            device,
            target_weights,
        )
        write_json(
            temporary / "training_history.json",
            {"observer": observer_history, "diagnostic": diagnostic_history},
        )
        _save_checkpoint(temporary / "model.pt", model, config, graph, targets)
        # All reported predictions come from a strict checkpoint reload.
        model = _reload_checkpoint(
            temporary / "model.pt", graph, targets, uncertainty, device
        )
        validation = _collect(
            model,
            _dataset(
                args,
                graph,
                targets,
                standardizer,
                ("validation_id",),
                evaluation=True,
            ),
            args,
            targets,
            device,
        )
        temperature = _fit_temperature(
            validation["probabilities"], validation["targets"]
        )
        validation["probabilities"] = _temperature_scale(
            validation["probabilities"], temperature
        )
        validation_anomaly_truth = np.asarray(
            [targets.code_is_fault[int(index)] for index in validation["targets"]],
            dtype=np.int64,
        )
        anomaly_threshold = _fit_anomaly_threshold(
            validation["anomaly_probability"], validation_anomaly_truth
        )
        block_calibration, block_audit = counterfactual_group_mode_block_max(
            validation["probabilities"],
            validation["targets"],
            validation["modes"],
            validation["energies"],
            validation["counterfactual_groups"],
            validation["plant_profiles"],
        )
        calibrator = ModeConformalCalibrator(
            alpha=args.alpha,
            ood_alpha=args.ood_alpha,
        ).fit(
            block_calibration["probabilities"],
            block_calibration["targets"],
            block_calibration["modes"],
            block_calibration["energies"],
            splits=["validation_id"] * len(block_calibration["targets"]),
        )
        calibration_state = calibrator.to_dict()
        calibration_state.update(
            {
                "class_temperature": temperature,
                "class_temperature_fit_split": "validation_id",
                "anomaly_threshold": anomaly_threshold,
                "anomaly_threshold_fit_split": "validation_id",
                "anomaly_threshold_method": "maximum validation row-F1; highest-threshold tie break",
                "conformal_calibration_audit": block_audit,
            }
        )
        write_json(temporary / "conformal_state.json", calibration_state)
        test = _collect(
            model,
            _dataset(
                args,
                graph,
                targets,
                standardizer,
                test_splits,
                evaluation=True,
            ),
            args,
            targets,
            device,
        )
        test["probabilities"] = _temperature_scale(
            test["probabilities"], temperature
        )
        metrics = _evaluate_and_write(
            test,
            calibrator,
            targets,
            temporary,
            anomaly_threshold,
        )
        metrics["validation_calibration"] = {
            "class_temperature": temperature,
            "anomaly_threshold": anomaly_threshold,
            "conformal_count": calibrator.calibration_count,
        }
        write_json(temporary / "metrics.json", metrics)
        manifest = {
            "trainer_version": TRAINER_VERSION,
            "model_version": MODEL_VERSION,
            "dataset_manifest_sha256": sha256_file(
                args.dataset / "dataset_manifest.json"
            ),
            "split_manifest_sha256": sha256_file(
                args.splits / "split_manifest.json"
            ),
            "cache_manifest_sha256": (
                sha256_file(args.cache / "cache_manifest.json")
                if args.cache is not None
                else None
            ),
            "input_contract": {
                "feature_set": args.feature_set,
                "features": list(graph.features),
                "forbidden_feature_intersection": [],
                "route_controls": list(CONTROL_FIELDS),
                "time_policy": "time_s orders windows only; it is never a numeric input",
                "plant_mode_policy": "ignored; route mode derives from observable relays",
            },
            "split_policy": {
                "training": ["train_id"],
                "observer_training_codes": ["N00", "M01", "M02"],
                "conformal_calibration": ["validation_id"],
                "evaluation": list(test_splits),
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
            },
            "provenance": {
                name: sha256_file(BASE_DIR / name)
                for name in (
                    "reference_pid.json",
                    "sensor_catalog.json",
                    "ml_contract.json",
                    "fault_asset_contract.json",
                    "train_flowtwin_guard.py",
                    "flowtwin_guard/graph.py",
                    "flowtwin_guard/data.py",
                    "flowtwin_guard/cache.py",
                    "flowtwin_guard/model.py",
                    "flowtwin_guard/conformal.py",
                )
            },
            "artifacts": [*ARTIFACTS, "checksums.sha256"],
            "claim_boundary": (
                "Research-only synthetic pre-validation model; no field accuracy, "
                "pasteurization efficacy, HACCP conformity, SAFE decision, or RELEASE "
                "authorization is claimed."
            ),
        }
        write_json(temporary / "run_manifest.json", manifest)
        write_checksums(temporary, ARTIFACTS)
        _commit_output(temporary, args.output, existed_empty)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return args.output.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    print(
        f"FlowTwin-Guard complete: {output}\n"
        f"test accuracy={metrics['class_diagnosis']['accuracy']:.6f}, "
        f"coverage={metrics['conformal']['empirical_coverage']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
