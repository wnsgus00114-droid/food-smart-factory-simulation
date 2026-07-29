"""Versioned, leakage-safe FlowTwin-Guard ablation adapters.

The canonical model deliberately remains unchanged.  This module constructs a
subclass with narrowly scoped input transforms or graph-module substitutions
for the nine ablations specified in :mod:`FLOWTWIN_GUARD.md`.  Training- and
evaluation-only changes are exposed as explicit protocol metadata so a runner
cannot silently claim an ablation while retaining the full protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .graph import CONTROL_FIELDS, ProcessGraph
from .model import (
    MODEL_VERSION,
    FlowTwinConfig,
    FlowTwinGuard,
    TransportDelayBlock,
)


ABLATION_VERSION = "0.2.0"

ABLATION_NAMES = (
    "no_delay",
    "fixed_nominal_delay",
    "static_route",
    "single_relation",
    "raw_only",
    "no_uncertainty",
    "no_counterfactual_loss",
    "no_conformal",
    "no_status_relays",
)
ABLATION_CHOICES = ("none", *ABLATION_NAMES)

STATUS_RELAY_FEATURES = (
    "power_good_signal",
    "temperature_sensor_quality_ok",
)

_BASE_SIGNATURE: dict[str, Any] = {
    "delay_mode": "flow_dependent",
    "route_mode": "dynamic_observable",
    "relation_mode": "dual_relation",
    "residual_mode": "uncertainty_normalized",
    "sensor_uncertainty_enabled": True,
    "counterfactual_loss_weight": 0.10,
    "conformal_enabled": True,
    "excluded_feature_names": (),
}

_SIGNATURE_OVERRIDES: dict[str, dict[str, Any]] = {
    "none": {},
    "no_delay": {"delay_mode": "zero"},
    "fixed_nominal_delay": {"delay_mode": "fixed_nominal"},
    "static_route": {"route_mode": "static_all_observable_edges"},
    "single_relation": {"relation_mode": "single_advective"},
    "raw_only": {"residual_mode": "raw_only"},
    "no_uncertainty": {
        "residual_mode": "unnormalized_twin_residual",
        "sensor_uncertainty_enabled": False,
    },
    "no_counterfactual_loss": {"counterfactual_loss_weight": 0.0},
    "no_conformal": {"conformal_enabled": False},
    "no_status_relays": {"excluded_feature_names": STATUS_RELAY_FEATURES},
}


@dataclass(frozen=True)
class AblationConfig:
    """Serializable experiment contract for exactly one ablation axis."""

    name: str = "none"
    ablation_version: str = ABLATION_VERSION
    base_model_version: str = MODEL_VERSION
    delay_mode: str = "flow_dependent"
    route_mode: str = "dynamic_observable"
    relation_mode: str = "dual_relation"
    residual_mode: str = "uncertainty_normalized"
    sensor_uncertainty_enabled: bool = True
    counterfactual_loss_weight: float = 0.10
    conformal_enabled: bool = True
    excluded_feature_names: tuple[str, ...] = ()
    fixed_nominal_flow_l_h: float = 20_000.0
    fixed_nominal_holding_time_s: float = 18.0
    fixed_sensor_to_fdv_delay_s: float = 1.5
    fixed_post_fdv_residence_time_s: float = 5.0

    def __post_init__(self) -> None:
        if self.ablation_version != ABLATION_VERSION:
            raise ValueError(
                f"unsupported ablation version {self.ablation_version!r}; "
                f"expected {ABLATION_VERSION!r}"
            )
        if self.base_model_version != MODEL_VERSION:
            raise ValueError(
                f"ablation targets FlowTwin model {self.base_model_version!r}, "
                f"but this code provides {MODEL_VERSION!r}"
            )
        if self.name not in ABLATION_CHOICES:
            raise ValueError(
                f"unknown ablation {self.name!r}; expected one of {ABLATION_CHOICES}"
            )
        for field_name in (
            "fixed_nominal_flow_l_h",
            "fixed_nominal_holding_time_s",
            "fixed_sensor_to_fdv_delay_s",
            "fixed_post_fdv_residence_time_s",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not math.isfinite(self.counterfactual_loss_weight) or (
            self.counterfactual_loss_weight < 0.0
        ):
            raise ValueError("counterfactual_loss_weight must be finite and non-negative")

        expected = dict(_BASE_SIGNATURE)
        expected.update(_SIGNATURE_OVERRIDES[self.name])
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(
                    f"{self.name!r} requires {field_name}={value!r}, got "
                    f"{getattr(self, field_name)!r}; use get_ablation_config()"
                )

    @property
    def training_options(self) -> dict[str, Any]:
        """Options the trainer must honour in addition to model construction."""

        return {
            "train_nominal_observer": self.residual_mode != "raw_only",
            "requires_counterfactual_reference": self.counterfactual_loss_weight > 0.0,
            "loss_weight_overrides": {
                "counterfactual": self.counterfactual_loss_weight,
                "delay_regularization": (
                    0.0 if self.delay_mode in {"zero", "fixed_nominal"} else 0.001
                ),
            },
        }

    @property
    def evaluation_options(self) -> dict[str, Any]:
        """Options the evaluator must honour for calibration/decision output."""

        return {
            "conformal_enabled": self.conformal_enabled,
            "abstention_enabled": self.conformal_enabled,
            "decision_policy": (
                "validation_only_conformal" if self.conformal_enabled else "argmax_only"
            ),
        }

    def assert_training_protocol(
        self,
        *,
        train_nominal_observer: bool,
        counterfactual_loss_weight: float,
    ) -> None:
        """Fail if a runner's effective training settings contradict the label."""

        expected = self.training_options
        if bool(train_nominal_observer) != expected["train_nominal_observer"]:
            raise ValueError(
                f"{self.name}: train_nominal_observer must be "
                f"{expected['train_nominal_observer']}"
            )
        if not math.isclose(
            float(counterfactual_loss_weight),
            self.counterfactual_loss_weight,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{self.name}: counterfactual loss weight must be "
                f"{self.counterfactual_loss_weight}"
            )

    def assert_evaluation_protocol(self, *, conformal_enabled: bool) -> None:
        """Fail if an evaluator's abstention policy contradicts the label."""

        if bool(conformal_enabled) != self.conformal_enabled:
            raise ValueError(
                f"{self.name}: conformal_enabled must be {self.conformal_enabled}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AblationConfig":
        data = dict(value)
        if "excluded_feature_names" in data:
            data["excluded_feature_names"] = tuple(data["excluded_feature_names"])
        return cls(**data)


def get_ablation_config(
    name: str,
    *,
    fixed_nominal_flow_l_h: float = 20_000.0,
    fixed_nominal_holding_time_s: float = 18.0,
    fixed_sensor_to_fdv_delay_s: float = 1.5,
    fixed_post_fdv_residence_time_s: float = 5.0,
) -> AblationConfig:
    """Return the canonical, versioned configuration for ``name``.

    ``full`` and ``baseline`` are accepted as CLI-friendly aliases for
    ``none`` but are canonicalised before serialization.
    """

    canonical = {"full": "none", "baseline": "none"}.get(name, name)
    if canonical not in ABLATION_CHOICES:
        raise ValueError(
            f"unknown ablation {name!r}; expected one of {ABLATION_CHOICES}"
        )
    values = dict(_BASE_SIGNATURE)
    values.update(_SIGNATURE_OVERRIDES[canonical])
    return AblationConfig(
        name=canonical,
        fixed_nominal_flow_l_h=fixed_nominal_flow_l_h,
        fixed_nominal_holding_time_s=fixed_nominal_holding_time_s,
        fixed_sensor_to_fdv_delay_s=fixed_sensor_to_fdv_delay_s,
        fixed_post_fdv_residence_time_s=fixed_post_fdv_residence_time_s,
        **values,
    )


class StaticRouteGate(nn.Module):
    """A static P&ID adjacency with only explicitly unobserved edges closed."""

    def __init__(self, graph: ProcessGraph) -> None:
        super().__init__()
        self.register_buffer(
            "static_edge_gate",
            torch.tensor(
                [edge.gate_kind != "unobserved" for edge in graph.edges],
                dtype=torch.bool,
            ),
        )

    def forward(self, controls_raw: Tensor) -> Tensor:
        if controls_raw.ndim != 3 or controls_raw.shape[-1] != len(CONTROL_FIELDS):
            raise ValueError(
                f"controls_raw must have shape [B,T,{len(CONTROL_FIELDS)}]"
            )
        if not torch.isfinite(controls_raw).all():
            raise ValueError("route controls must be finite")
        return self.static_edge_gate.to(controls_raw.dtype).view(1, 1, -1).expand(
            controls_raw.shape[0], controls_raw.shape[1], -1
        )


class _FixedNominalDelayBlock(TransportDelayBlock):
    """Hold only the V/Q denominator fixed while preserving observed routes.

    RouteGate must continue to see the measured flow.  Replacing the flow in
    the shared controls tensor would turn production/common route edges on at
    zero flow and would confound the delay ablation with a route ablation.
    """

    def __init__(
        self,
        graph: ProcessGraph,
        hidden_dim: int,
        dropout: float,
        maximum_prior_correction_fraction: float,
        fixed_nominal_flow_l_h: float,
    ) -> None:
        super().__init__(
            graph,
            hidden_dim,
            dropout,
            maximum_prior_correction_fraction,
        )
        self.register_buffer(
            "fixed_nominal_flow_l_h",
            torch.tensor(float(fixed_nominal_flow_l_h), dtype=torch.float32),
            persistent=False,
        )

    def _delayed_sources(
        self,
        advective: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        delay_controls = controls_raw.clone()
        delay_controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = (
            self.fixed_nominal_flow_l_h.to(
                dtype=controls_raw.dtype, device=controls_raw.device
            )
        )
        return super()._delayed_sources(
            advective, delay_controls, dt_s, edge_volume_l
        )


class AblatedFlowTwinGuard(FlowTwinGuard):
    """FlowTwin-Guard with one audited ablation applied outside ``model.py``."""

    def __init__(
        self,
        graph: ProcessGraph,
        config: FlowTwinConfig,
        sensor_uncertainty_normalized: Tensor,
        ablation_config: AblationConfig,
    ) -> None:
        if not isinstance(ablation_config, AblationConfig):
            raise TypeError("ablation_config must be an AblationConfig")
        if (
            ablation_config.name == "no_uncertainty"
            and not torch.any(sensor_uncertainty_normalized > 0.0)
        ):
            raise ValueError(
                "no_uncertainty would be a no-op because the supplied uncertainty "
                "vector contains no positive values"
            )
        missing_relays = set(ablation_config.excluded_feature_names) - set(graph.features)
        if missing_relays:
            raise ValueError(
                f"{ablation_config.name} requires feature(s) absent from the graph: "
                f"{sorted(missing_relays)}"
            )

        effective_uncertainty = sensor_uncertainty_normalized
        if not ablation_config.sensor_uncertainty_enabled:
            effective_uncertainty = torch.zeros_like(sensor_uncertainty_normalized)
        super().__init__(graph, config, effective_uncertainty)
        self.ablation_config = ablation_config
        self._feature_index = {name: index for index, name in enumerate(graph.features)}

        fixed_profile = {
            "nominal_flow_l_h": ablation_config.fixed_nominal_flow_l_h,
            "nominal_holding_time_s": ablation_config.fixed_nominal_holding_time_s,
            "sensor_to_fdv_delay_s": ablation_config.fixed_sensor_to_fdv_delay_s,
            "post_fdv_residence_time_s": (
                ablation_config.fixed_post_fdv_residence_time_s
            ),
        }
        self.register_buffer(
            "fixed_nominal_edge_volume_l",
            torch.tensor(
                graph.edge_volumes_for_profile(fixed_profile), dtype=torch.float32
            ),
        )

        with torch.no_grad():
            if ablation_config.delay_mode == "zero":
                for block in self.graph_blocks:
                    # L-011 is normally forced to one sample regardless of volume.
                    # It must also be disabled for a literal tau=0 ablation.
                    block.one_step_edge.zero_()
            elif ablation_config.delay_mode == "fixed_nominal":
                # Recreate blocks with an overridden delay calculation, then
                # restore the RNG state so this mechanical substitution does
                # not alter dropout/random streams relative to the full model.
                rng_state = torch.random.get_rng_state()
                replacements: list[TransportDelayBlock] = []
                for block in self.graph_blocks:
                    replacement = _FixedNominalDelayBlock(
                        graph,
                        config.hidden_dim,
                        config.dropout,
                        config.maximum_prior_correction_fraction,
                        ablation_config.fixed_nominal_flow_l_h,
                    )
                    replacement.load_state_dict(block.state_dict(), strict=True)
                    replacement.max_correction = 0.0
                    replacements.append(replacement)
                self.graph_blocks = nn.ModuleList(replacements)
                torch.random.set_rng_state(rng_state)

            if ablation_config.relation_mode == "single_advective":
                # Every feature and every observable edge now follows the same
                # V/Q-delayed material relation.  The instantaneous branch is
                # deliberately discarded in _single_relation_forward().
                self.projector.advective_feature.fill_(True)
                for block in self.graph_blocks:
                    block.advective_edge.fill_(True)

        if ablation_config.route_mode == "static_all_observable_edges":
            for block in self.graph_blocks:
                block.route_gate = StaticRouteGate(graph)

    @property
    def training_options(self) -> dict[str, Any]:
        return self.ablation_config.training_options

    @property
    def evaluation_options(self) -> dict[str, Any]:
        return self.ablation_config.evaluation_options

    def _validate_forward_contract(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> None:
        if x.ndim != 3 or x.shape[-1] != self.config.feature_count:
            raise ValueError("x must have shape [B,T,feature_count]")
        if controls_raw.shape != (*x.shape[:2], len(CONTROL_FIELDS)):
            raise ValueError(
                f"controls_raw must have shape [B,T,{len(CONTROL_FIELDS)}]"
            )
        if dt_s.shape != x.shape[:2] or valid_mask.shape != x.shape[:2]:
            raise ValueError("dt_s and valid_mask must share x batch/time dimensions")
        if edge_volume_l.shape != (x.shape[0], self.config.edge_count):
            raise ValueError("edge_volume_l must have shape [B,edge_count]")
        if not torch.isfinite(x).all() or not torch.isfinite(controls_raw).all():
            raise ValueError("model inputs must be finite")
        if not torch.isfinite(dt_s).all() or torch.any(dt_s <= 0.0):
            raise ValueError("dt_s must be finite and positive")
        if not torch.isfinite(edge_volume_l).all() or torch.any(edge_volume_l < 0.0):
            raise ValueError("edge volumes must be finite and non-negative")

    def transform_inputs(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply auditable input-level transforms without mutating the batch."""

        self._validate_forward_contract(
            x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        transformed_x = x
        transformed_controls = controls_raw
        transformed_volume = edge_volume_l

        if self.ablation_config.excluded_feature_names:
            transformed_x = x.clone()
            relay_indices = [
                self._feature_index[name]
                for name in self.ablation_config.excluded_feature_names
            ]
            transformed_x[..., relay_indices] = 0.0
            transformed_controls = controls_raw.clone()
            # Absence of the direct power relay must not itself appear as a
            # power-failure label through RouteGate.  The neutral graph prior
            # is therefore "power available"; physical waveforms remain.
            transformed_controls[..., CONTROL_FIELDS.index("power_good_signal")] = 1.0

        if self.ablation_config.delay_mode == "zero":
            transformed_volume = torch.zeros_like(edge_volume_l)
        elif self.ablation_config.delay_mode == "fixed_nominal":
            transformed_volume = self.fixed_nominal_edge_volume_l.to(
                dtype=edge_volume_l.dtype, device=edge_volume_l.device
            ).view(1, -1).expand(x.shape[0], -1)

        return (
            transformed_x,
            transformed_controls,
            dt_s,
            transformed_volume,
            valid_mask,
        )

    def transform_observer_input(self, x: Tensor) -> Tensor:
        """Apply the same feature exclusion during pretraining and inference."""

        if not self.ablation_config.excluded_feature_names:
            return x
        transformed = x.clone()
        relay_indices = [
            self._feature_index[name]
            for name in self.ablation_config.excluded_feature_names
        ]
        transformed[..., relay_indices] = 0.0
        return transformed

    def observer_outputs(
        self, x: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.transform_observer_input(x)
        mode = self.ablation_config.residual_mode
        if mode == "uncertainty_normalized":
            return super().observer_outputs(x, valid_mask)

        observer_valid = valid_mask.clone()
        if observer_valid.shape[1] > 0:
            observer_valid[:, 0] = False
        if mode == "raw_only":
            zero = torch.zeros_like(x)
            return zero, zero, zero, observer_valid
        if mode == "unnormalized_twin_residual":
            mean, log_variance, _ = self.observer(x, valid_mask)
            residual = (x - mean) * observer_valid.unsqueeze(-1).to(x.dtype)
            return mean, log_variance, residual, observer_valid
        raise RuntimeError(f"unsupported residual mode {mode!r}")

    def observer_nll(
        self, x: Tensor, output: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tensor:
        x = self.transform_observer_input(x)
        if self.ablation_config.residual_mode == "raw_only":
            # A differentiable zero keeps generic training loops operational,
            # although compliant runners skip observer pretraining entirely.
            return next(self.observer.parameters()).sum() * 0.0
        if self.ablation_config.residual_mode == "unnormalized_twin_residual":
            # Removing the Twin uncertainty denominator also removes its
            # heteroscedastic weighting from observer fitting.  Otherwise the
            # variance head could still change the learned nominal mean even
            # though variance is hidden from the diagnosis network.
            per_feature = (x - output["nominal_mean"]).square()
            mask = valid_mask & output["observer_valid"]
            if not torch.any(mask):
                return per_feature.sum() * 0.0
            return per_feature[mask].mean()
        return super().observer_nll(x, output, valid_mask)

    def delay_regularization(self) -> Tensor:
        if self.ablation_config.delay_mode in {"zero", "fixed_nominal"}:
            return self.graph_blocks[0].prior_delay_parameter.sum() * 0.0
        return super().delay_regularization()

    def _single_relation_forward(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
        *,
        include_node_embeddings: bool,
    ) -> dict[str, Tensor]:
        mean, log_variance, residual, observer_valid = self.observer_outputs(
            x, valid_mask
        )
        advective, _unused_control = self.projector(x, residual, valid_mask)
        advective = self.advective_temporal(advective, valid_mask)
        control = torch.zeros_like(advective)
        transport_errors: list[Tensor] = []
        delay_seconds: Tensor | None = None
        for block in self.graph_blocks:
            advective, _unused_control, transport_error, delay_seconds = block(
                advective,
                control,
                controls_raw,
                dt_s,
                edge_volume_l,
                valid_mask,
            )
            control = torch.zeros_like(advective)
            transport_errors.append(transport_error)

        node_embedding = self.fusion(torch.cat([advective, control], dim=-1))
        attention = torch.softmax(
            self.node_attention(node_embedding).squeeze(-1), dim=-1
        )
        pooled = (attention.unsqueeze(-1) * node_embedding).sum(dim=2)
        pooled = pooled * valid_mask.unsqueeze(-1).to(pooled.dtype)
        class_logits = self.class_head(pooled)
        result = {
            "class_logits": class_logits,
            "anomaly_logits": self.anomaly_head(pooled),
            "location_logits": self.location_head(pooled),
            "mechanism_logits": self.mechanism_head(pooled),
            "ood_energy": -torch.logsumexp(class_logits, dim=-1),
            "embedding": pooled,
            "nominal_mean": mean,
            "nominal_log_variance": log_variance,
            "residual": residual,
            "observer_valid": observer_valid,
            "transport_error": torch.stack(transport_errors).mean(dim=0),
            "delay_seconds": (
                delay_seconds
                if delay_seconds is not None
                else torch.zeros_like(edge_volume_l[:, None, :])
            ),
        }
        if include_node_embeddings:
            result["node_embedding"] = node_embedding
            result["node_attention"] = attention
        return result

    def forward(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
        *,
        include_node_embeddings: bool = False,
    ) -> dict[str, Tensor]:
        x, controls_raw, dt_s, edge_volume_l, valid_mask = self.transform_inputs(
            x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        if self.ablation_config.relation_mode == "single_advective":
            return self._single_relation_forward(
                x,
                controls_raw,
                dt_s,
                edge_volume_l,
                valid_mask,
                include_node_embeddings=include_node_embeddings,
            )
        return super().forward(
            x,
            controls_raw,
            dt_s,
            edge_volume_l,
            valid_mask,
            include_node_embeddings=include_node_embeddings,
        )


def build_ablation_model(
    graph: ProcessGraph,
    model_config: FlowTwinConfig,
    sensor_uncertainty_normalized: Tensor,
    ablation: str | AblationConfig = "none",
) -> AblatedFlowTwinGuard:
    """Construct a model whose state and protocol identify the ablation."""

    resolved = get_ablation_config(ablation) if isinstance(ablation, str) else ablation
    if not isinstance(resolved, AblationConfig):
        raise TypeError("ablation must be a name or AblationConfig")
    return AblatedFlowTwinGuard(
        graph,
        model_config,
        sensor_uncertainty_normalized,
        resolved,
    )


__all__ = [
    "ABLATION_CHOICES",
    "ABLATION_NAMES",
    "ABLATION_VERSION",
    "STATUS_RELAY_FEATURES",
    "AblatedFlowTwinGuard",
    "AblationConfig",
    "StaticRouteGate",
    "build_ablation_model",
    "get_ablation_config",
]
