"""Causal TCN--FlowTwin hybrid research candidate.

The candidate combines two complementary representations without changing the
benchmark input contract:

* a causal TCN learns discriminative temporal patterns from the standardized
  sensor observations;
* the FlowTwin branch uses a persistence-skip nominal observer, normalized
  residuals, route gates, and fractional ``V/Q`` transport delays;
* a pointwise sigmoid gate forms a causal convex fusion of both branches; and
* a taxonomy-aware hierarchical head makes the binary anomaly probability
  exactly coherent with the 20-way class probability.

The fault/non-fault partition is serialized from ``TargetContract.code_is_fault``;
this matters because N00, M01, and M02 are all non-fault behaviours.  For every
valid sample the head therefore satisfies ``sum(p(fault classes)) = p_fault``
and ``sum(p(non-fault classes)) = 1 - p_fault``.  PLC, HACCP, release,
simulator-truth, and future-sample signals are deliberately absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph import CONTROL_FIELDS, ProcessGraph
from .model import FeatureNodeProjector, NodeTemporalEncoder, TransportDelayBlock


HYBRID_MODEL_VERSION = "0.1.0"


@dataclass(frozen=True)
class FlowTwinHybridConfig:
    """Strict, JSON-round-trippable configuration for :class:`FlowTwinHybrid`."""

    feature_count: int
    node_count: int
    edge_count: int
    class_count: int
    fault_class_indices: tuple[int, ...]
    location_count: int
    mechanism_count: int
    hidden_dim: int = 32
    temporal_layers: int = 2
    graph_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.10
    observer_hidden_dim: int = 48
    maximum_prior_correction_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "feature_count",
            "node_count",
            "edge_count",
            "class_count",
            "location_count",
            "mechanism_count",
            "hidden_dim",
            "temporal_layers",
            "graph_layers",
            "kernel_size",
            "observer_hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.class_count < 2:
            raise ValueError("class_count must include at least two classes")
        fault_indices = tuple(self.fault_class_indices)
        if not fault_indices:
            raise ValueError("fault_class_indices must contain at least one class")
        if len(set(fault_indices)) != len(fault_indices):
            raise ValueError("fault_class_indices must be unique")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.class_count
            for index in fault_indices
        ):
            raise ValueError("fault_class_indices are outside the class range")
        if len(fault_indices) == self.class_count:
            raise ValueError("at least one non-fault class is required")
        object.__setattr__(self, "fault_class_indices", fault_indices)
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ValueError("dropout must be numeric")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        correction = self.maximum_prior_correction_fraction
        if not isinstance(correction, (int, float)) or isinstance(correction, bool):
            raise ValueError("maximum_prior_correction_fraction must be numeric")
        if not 0.0 <= float(correction) <= 0.5:
            raise ValueError(
                "maximum_prior_correction_fraction must be in [0, 0.5]"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned configuration as JSON-safe values."""

        return {"hybrid_model_version": HYBRID_MODEL_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FlowTwinHybridConfig":
        """Load a configuration, rejecting missing, unknown, or stale fields."""

        if not isinstance(value, Mapping):
            raise TypeError("hybrid configuration must be a mapping")
        data = dict(value)
        expected = {field.name for field in fields(cls)} | {"hybrid_model_version"}
        missing = expected - set(data)
        unknown = set(data) - expected
        if missing:
            raise ValueError(f"missing hybrid configuration fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown hybrid configuration fields: {sorted(unknown)}")
        version = data.pop("hybrid_model_version")
        if version != HYBRID_MODEL_VERSION:
            raise ValueError(f"unsupported hybrid model version {version!r}")
        data["fault_class_indices"] = tuple(data["fault_class_indices"])
        return cls(**data)


def _validate_inputs(
    config: FlowTwinHybridConfig,
    x: Tensor,
    controls_raw: Tensor,
    dt_s: Tensor,
    edge_volume_l: Tensor,
    valid_mask: Tensor,
) -> None:
    if x.ndim != 3 or x.shape[-1] != config.feature_count:
        raise ValueError(
            f"x must have shape [B,T,{config.feature_count}], got {tuple(x.shape)}"
        )
    batch, steps = x.shape[:2]
    if controls_raw.shape != (batch, steps, len(CONTROL_FIELDS)):
        raise ValueError(
            "controls_raw must have shape "
            f"[B,T,{len(CONTROL_FIELDS)}]"
        )
    if dt_s.shape != (batch, steps):
        raise ValueError("dt_s must have shape [B,T]")
    if edge_volume_l.shape != (batch, config.edge_count):
        raise ValueError(f"edge_volume_l must have shape [B,{config.edge_count}]")
    if valid_mask.shape != (batch, steps) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean [B,T] tensor")
    for name, value in (
        ("x", x),
        ("controls_raw", controls_raw),
        ("dt_s", dt_s),
        ("edge_volume_l", edge_volume_l),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if torch.any(dt_s <= 0.0):
        raise ValueError("dt_s must be positive")
    if torch.any(edge_volume_l < 0.0):
        raise ValueError("edge_volume_l must be non-negative")
    flow = controls_raw[..., CONTROL_FIELDS.index("measured_flow_l_h")]
    if torch.any(flow < 0.0):
        raise ValueError("measured flow cannot be negative")
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("every sequence must contain at least one valid sample")


class PersistenceSkipObserver(nn.Module):
    """Predict a delta around the strictly previous observation.

    The recurrent network at index ``t`` receives ``x[t-1]``.  Its mean is
    ``x[t-1] + delta[t]``; the zero-initialized delta head consequently starts
    as the persistence model while remaining fully trainable.  A prediction is
    valid only when both the current and immediately previous samples are
    valid.
    """

    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.gru = nn.GRU(feature_count, hidden_dim, batch_first=True)
        self.delta_head = nn.Linear(hidden_dim, feature_count)
        self.log_variance_head = nn.Linear(hidden_dim, feature_count)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.feature_count:
            raise ValueError("observer input has the wrong feature width")
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("observer valid_mask must be a boolean [B,T] tensor")
        masked = x * valid_mask.unsqueeze(-1).to(x.dtype)
        previous = torch.zeros_like(masked)
        previous[:, 1:] = masked[:, :-1]
        hidden, _ = self.gru(previous)
        mean = previous + self.delta_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 4.0)
        observer_valid = valid_mask.clone()
        observer_valid[:, 0] = False
        observer_valid[:, 1:] &= valid_mask[:, :-1]
        mask = valid_mask.unsqueeze(-1).to(x.dtype)
        return mean * mask, log_variance * mask, observer_valid


class _CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.convolution(F.pad(values, (self.left_padding, 0)))


class _CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convolution_1 = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.convolution_2 = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        residual = values
        hidden = self.convolution_1(values.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(F.gelu(self.norm_1(hidden)))
        hidden = self.convolution_2(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = residual + self.dropout(F.gelu(self.norm_2(hidden)))
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)


class _CausalTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _CausalTemporalBlock(
                    hidden_dim,
                    kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        mask = valid_mask.unsqueeze(-1).to(x.dtype)
        hidden = self.input_projection(x * mask) * mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        return self.output_norm(hidden) * mask
class FlowTwinHybrid(nn.Module):
    """Gated causal fusion of a discriminative TCN and the FlowTwin branch."""

    def __init__(
        self,
        graph: ProcessGraph,
        config: FlowTwinHybridConfig,
        sensor_uncertainty_normalized: Tensor,
    ) -> None:
        super().__init__()
        if (
            config.feature_count != len(graph.features)
            or config.node_count != len(graph.nodes)
            or config.edge_count != len(graph.edges)
        ):
            raise ValueError("model config dimensions do not match process graph")
        if sensor_uncertainty_normalized.shape != (config.feature_count,):
            raise ValueError("sensor uncertainty vector has the wrong width")
        if not torch.isfinite(sensor_uncertainty_normalized).all() or torch.any(
            sensor_uncertainty_normalized < 0.0
        ):
            raise ValueError("sensor uncertainty must be finite and non-negative")

        self.config = config
        fault_indices = tuple(config.fault_class_indices)
        non_fault_indices = tuple(
            index for index in range(config.class_count) if index not in fault_indices
        )
        self.register_buffer(
            "fault_class_indices",
            torch.tensor(fault_indices, dtype=torch.long),
        )
        self.register_buffer(
            "non_fault_class_indices",
            torch.tensor(non_fault_indices, dtype=torch.long),
        )
        self.observer = PersistenceSkipObserver(
            config.feature_count, config.observer_hidden_dim
        )

        # Discriminative branch: identical information boundary to the raw TCN
        # baseline, but trained jointly with the physics branch.
        self.temporal_branch = _CausalTCN(
            config.feature_count,
            config.hidden_dim,
            config.temporal_layers,
            config.kernel_size,
            config.dropout,
        )

        # Physics/residual branch: reuse the tested FlowTwin P&ID components.
        self.projector = FeatureNodeProjector(graph, config.hidden_dim)
        self.advective_temporal = NodeTemporalEncoder(config.hidden_dim)
        self.control_temporal = NodeTemporalEncoder(config.hidden_dim)
        self.graph_blocks = nn.ModuleList(
            [
                TransportDelayBlock(
                    graph,
                    config.hidden_dim,
                    config.dropout,
                    config.maximum_prior_correction_fraction,
                )
                for _ in range(config.graph_layers)
            ]
        )
        self.physics_node_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.node_attention = nn.Linear(config.hidden_dim, 1)

        # The sigmoid is pointwise in time.  The convex mixture makes gate
        # values directly interpretable as TCN contribution weights.
        self.fusion_gate_head = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)

        self.fault_logit_head = nn.Linear(config.hidden_dim, 1)
        self.conditional_non_fault_head = nn.Linear(
            config.hidden_dim, len(non_fault_indices)
        )
        self.conditional_fault_head = nn.Linear(
            config.hidden_dim, len(fault_indices)
        )
        self.location_head = nn.Linear(config.hidden_dim, config.location_count)
        self.mechanism_head = nn.Linear(config.hidden_dim, config.mechanism_count)
        self.register_buffer(
            "sensor_uncertainty_normalized",
            sensor_uncertainty_normalized.detach().clone().to(torch.float32),
        )

    def set_observer_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze only the nominal observer for two-stage training."""

        for parameter in self.observer.parameters():
            parameter.requires_grad_(trainable)

    def observer_outputs(
        self, x: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, log_variance, observer_valid = self.observer(x, valid_mask)
        variance = torch.exp(log_variance) + self.sensor_uncertainty_normalized.square().view(
            1, 1, -1
        )
        residual = (x - mean) / torch.sqrt(variance.clamp_min(1e-6))
        residual = residual * observer_valid.unsqueeze(-1).to(residual.dtype)
        return mean, log_variance, residual, observer_valid

    def _physics_embedding(
        self,
        x: Tensor,
        residual: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        advective, control = self.projector(x, residual, valid_mask)
        advective = self.advective_temporal(advective, valid_mask)
        control = self.control_temporal(control, valid_mask)
        transport_errors: list[Tensor] = []
        delay_seconds: Tensor | None = None
        for block in self.graph_blocks:
            advective, control, error, delay_seconds = block(
                advective,
                control,
                controls_raw,
                dt_s,
                edge_volume_l,
                valid_mask,
            )
            transport_errors.append(error)
        node_embedding = self.physics_node_fusion(
            torch.cat([advective, control], dim=-1)
        )
        node_attention = torch.softmax(
            self.node_attention(node_embedding).squeeze(-1), dim=-1
        )
        pooled = (node_attention.unsqueeze(-1) * node_embedding).sum(dim=2)
        mask = valid_mask.unsqueeze(-1).to(pooled.dtype)
        pooled = pooled * mask
        if delay_seconds is None:  # graph_layers is validated positive; defensive only.
            delay_seconds = torch.zeros_like(edge_volume_l[:, None, :])
        return (
            pooled,
            node_embedding,
            node_attention,
            torch.stack(transport_errors).mean(dim=0),
            delay_seconds,
        )

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
        _validate_inputs(
            self.config, x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        mean, log_variance, residual, observer_valid = self.observer_outputs(
            x, valid_mask
        )
        temporal_embedding = self.temporal_branch(x, valid_mask)
        (
            physics_embedding,
            node_embedding,
            node_attention,
            transport_error,
            delay_seconds,
        ) = self._physics_embedding(
            x,
            residual,
            controls_raw,
            dt_s,
            edge_volume_l,
            valid_mask,
        )

        gate = torch.sigmoid(
            self.fusion_gate_head(
                torch.cat([temporal_embedding, physics_embedding], dim=-1)
            )
        )
        fused = self.fusion_norm(
            gate * temporal_embedding + (1.0 - gate) * physics_embedding
        )
        output_mask = valid_mask.unsqueeze(-1).to(fused.dtype)
        fused = fused * output_mask
        gate = gate * output_mask

        # Binary anomaly evidence and conditional non-N00 evidence are composed
        # in log space.  A shared negative residual/transport score preserves
        # the exact softmax probabilities while making energy increase with a
        # physically meaningful departure from the learned nominal process.
        fault_logit = self.fault_logit_head(fused).squeeze(-1)
        conditional_non_fault_logits = self.conditional_non_fault_head(fused)
        conditional_fault_logits = self.conditional_fault_head(fused)
        log_non_fault = F.logsigmoid(-fault_logit).unsqueeze(-1) + F.log_softmax(
            conditional_non_fault_logits, dim=-1
        )
        log_fault = F.logsigmoid(fault_logit).unsqueeze(-1) + F.log_softmax(
            conditional_fault_logits, dim=-1
        )
        class_parts: list[Tensor] = []
        non_fault_position = {
            int(index): position
            for position, index in enumerate(self.non_fault_class_indices.tolist())
        }
        fault_position = {
            int(index): position
            for position, index in enumerate(self.fault_class_indices.tolist())
        }
        for class_index in range(self.config.class_count):
            if class_index in fault_position:
                class_parts.append(log_fault[..., fault_position[class_index]])
            else:
                class_parts.append(log_non_fault[..., non_fault_position[class_index]])
        class_log_probability = torch.stack(class_parts, dim=-1)
        residual_energy = torch.log1p(residual.square().mean(dim=-1))
        transport_energy = torch.log1p(transport_error.clamp_min(0.0))
        physical_ood_score = (residual_energy + transport_energy) * valid_mask.to(
            fused.dtype
        )
        class_logits = class_log_probability - physical_ood_score.unsqueeze(-1)
        anomaly_logits = torch.stack([torch.zeros_like(fault_logit), fault_logit], dim=-1)

        scalar_mask = valid_mask.to(fused.dtype)
        class_logits = class_logits * output_mask
        anomaly_logits = anomaly_logits * output_mask
        location_logits = self.location_head(fused) * output_mask
        mechanism_logits = self.mechanism_head(fused) * output_mask
        ood_energy = -torch.logsumexp(class_logits, dim=-1)
        ood_energy = ood_energy * scalar_mask

        result = {
            # Required six-output diagnostic contract.
            "class_logits": class_logits,
            "anomaly_logits": anomaly_logits,
            "location_logits": location_logits,
            "mechanism_logits": mechanism_logits,
            "embedding": fused,
            "ood_energy": ood_energy,
            # Observer/physics/fusion audit outputs.
            "nominal_mean": mean,
            "nominal_log_variance": log_variance,
            "residual": residual,
            "observer_valid": observer_valid,
            "transport_error": transport_error,
            "delay_seconds": delay_seconds,
            "temporal_embedding": temporal_embedding,
            "physics_embedding": physics_embedding,
            "fusion_gate": gate,
            "conditional_non_fault_logits": conditional_non_fault_logits * output_mask,
            "conditional_fault_logits": conditional_fault_logits * output_mask,
            "residual_energy": residual_energy * scalar_mask,
            "transport_energy": transport_energy * scalar_mask,
        }
        if include_node_embeddings:
            result["node_embedding"] = node_embedding
            result["node_attention"] = node_attention
        return result

    def observer_nll(
        self, x: Tensor, output: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tensor:
        """Gaussian nominal-observer loss for healthy training rows only."""

        variance = torch.exp(output["nominal_log_variance"])
        variance = variance + self.sensor_uncertainty_normalized.square().view(
            1, 1, -1
        )
        per_feature = 0.5 * (
            (x - output["nominal_mean"]).square() / variance.clamp_min(1e-6)
            + torch.log(variance.clamp_min(1e-6))
        )
        selected = valid_mask & output["observer_valid"]
        if not torch.any(selected):
            return per_feature.sum() * 0.0
        return per_feature[selected].mean()

    def delay_regularization(self) -> Tensor:
        """Bounded FlowTwin prior-delay correction penalty."""

        return torch.stack(
            [block.delay_regularization() for block in self.graph_blocks]
        ).mean()


__all__ = [
    "HYBRID_MODEL_VERSION",
    "FlowTwinHybrid",
    "FlowTwinHybridConfig",
    "PersistenceSkipObserver",
]
