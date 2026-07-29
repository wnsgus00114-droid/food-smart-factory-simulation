"""FlowTwin-Guard PyTorch architecture.

The model has four deliberately separated responsibilities:

1. a causal nominal observer trained only on healthy operating modes;
2. uncertainty-normalised residual construction;
3. dual-relation P&ID propagation (advective delay + instantaneous control);
4. hierarchical diagnosis heads with an energy score for OOD calibration.

PLC/HACCP decisions are not inputs to this module and remain an independent
safety layer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph import CONTROL_FIELDS, ProcessGraph


MODEL_VERSION = "0.1.0"


@dataclass(frozen=True)
class FlowTwinConfig:
    feature_count: int
    node_count: int
    edge_count: int
    class_count: int
    location_count: int
    mechanism_count: int
    hidden_dim: int = 32
    graph_layers: int = 2
    dropout: float = 0.10
    observer_hidden_dim: int = 48
    maximum_prior_correction_fraction: float = 0.25

    def __post_init__(self) -> None:
        integer_fields = (
            "feature_count",
            "node_count",
            "edge_count",
            "class_count",
            "location_count",
            "mechanism_count",
            "hidden_dim",
            "graph_layers",
            "observer_hidden_dim",
        )
        for field in integer_fields:
            if int(getattr(self, field)) < 1:
                raise ValueError(f"{field} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.maximum_prior_correction_fraction <= 0.5:
            raise ValueError("maximum_prior_correction_fraction must be in [0, 0.5]")

    def to_dict(self) -> dict[str, Any]:
        return {"model_version": MODEL_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FlowTwinConfig":
        data = dict(value)
        version = data.pop("model_version", MODEL_VERSION)
        if version != MODEL_VERSION:
            raise ValueError(f"unsupported FlowTwin model version {version!r}")
        return cls(**data)


def fractional_delay(history: Tensor, delay_steps: Tensor) -> tuple[Tensor, Tensor]:
    """Causally sample ``history[t-delay]`` with linear interpolation.

    Parameters
    ----------
    history:
        ``[batch, time, hidden]`` tensor.
    delay_steps:
        ``[batch, time]`` non-negative delay in sample intervals.

    Samples before the left boundary are returned as zero with ``valid=False``;
    they are never clamped to the first observed value.
    """

    if history.ndim != 3 or delay_steps.ndim != 2:
        raise ValueError("fractional_delay expects [B,T,H] and [B,T]")
    if history.shape[:2] != delay_steps.shape:
        raise ValueError("history and delay_steps batch/time dimensions differ")
    if not torch.isfinite(delay_steps).all():
        raise ValueError("delay_steps must be finite")
    if torch.any(delay_steps < 0.0):
        raise ValueError("delay_steps must be non-negative")
    batch, steps, hidden = history.shape
    query = (
        torch.arange(steps, device=history.device, dtype=delay_steps.dtype)
        .unsqueeze(0)
        .expand(batch, -1)
        - delay_steps
    )
    valid = query >= 0.0
    lower = torch.floor(query)
    fraction = (query - lower).to(history.dtype)
    lower_index = lower.to(torch.long).clamp(0, max(0, steps - 1))
    upper_index = (lower_index + 1).clamp(0, max(0, steps - 1))
    lower_values = torch.gather(
        history, 1, lower_index.unsqueeze(-1).expand(-1, -1, hidden)
    )
    upper_values = torch.gather(
        history, 1, upper_index.unsqueeze(-1).expand(-1, -1, hidden)
    )
    result = lower_values * (1.0 - fraction.unsqueeze(-1)) + upper_values * fraction.unsqueeze(-1)
    result = result * valid.unsqueeze(-1).to(result.dtype)
    return result, valid


class RouteGate(nn.Module):
    """Compute all edge gates from six allowlisted observable signals."""

    KINDS = (
        "production",
        "common",
        "forward",
        "divert",
        "cip",
        "steam",
        "cooling",
        "unobserved",
    )

    def __init__(self, graph: ProcessGraph) -> None:
        super().__init__()
        kind_index = {name: index for index, name in enumerate(self.KINDS)}
        self.register_buffer(
            "edge_gate_kind",
            torch.tensor([kind_index[edge.gate_kind] for edge in graph.edges]),
        )

    def forward(self, controls_raw: Tensor) -> Tensor:
        if controls_raw.ndim != 3 or controls_raw.shape[-1] != len(CONTROL_FIELDS):
            raise ValueError(
                f"controls_raw must have shape [B,T,{len(CONTROL_FIELDS)}]"
            )
        if not torch.isfinite(controls_raw).all():
            raise ValueError("route controls must be finite")
        flow, cip, fdv_feedback, steam, power, _chemical = controls_raw.unbind(-1)
        if torch.any(flow < 0.0):
            raise ValueError("measured flow cannot be negative")
        q_on = (flow > 1.0).to(controls_raw.dtype)
        cip = cip.clamp(0.0, 1.0)
        fdv_feedback = fdv_feedback.clamp(0.0, 1.0)
        steam = steam.clamp(0.0, 1.0)
        power = power.clamp(0.0, 1.0)
        production = (1.0 - cip) * q_on
        common = q_on
        forward = production * fdv_feedback
        divert = production * (1.0 - fdv_feedback)
        cip_gate = cip * q_on
        steam_gate = power * steam
        cooling = forward * power
        unobserved = torch.zeros_like(flow)
        by_kind = torch.stack(
            [
                production,
                common,
                forward,
                divert,
                cip_gate,
                steam_gate,
                cooling,
                unobserved,
            ],
            dim=-1,
        )
        return by_kind.index_select(-1, self.edge_gate_kind)


class NominalObserver(nn.Module):
    """One-step causal normal-behaviour observer.

    At time ``t`` the recurrent network receives the observation from ``t-1``.
    Consequently its nominal prediction cannot copy the current anomaly.
    """

    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.gru = nn.GRU(feature_count, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, feature_count)
        self.log_variance_head = nn.Linear(hidden_dim, feature_count)

    def forward(self, x: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.feature_count:
            raise ValueError("observer input has the wrong feature width")
        if valid_mask.shape != x.shape[:2]:
            raise ValueError("observer valid_mask shape mismatch")
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        hidden, _ = self.gru(shifted)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 4.0)
        observer_valid = valid_mask.clone()
        observer_valid[:, 0] = False
        mean = mean * valid_mask.unsqueeze(-1).to(mean.dtype)
        log_variance = log_variance * valid_mask.unsqueeze(-1).to(log_variance.dtype)
        return mean, log_variance, observer_valid


class FeatureNodeProjector(nn.Module):
    """Scatter uncertainty-aware feature pairs onto their P&ID assets."""

    def __init__(self, graph: ProcessGraph, hidden_dim: int) -> None:
        super().__init__()
        self.node_count = len(graph.nodes)
        self.feature_count = len(graph.features)
        self.hidden_dim = hidden_dim
        self.value_projection = nn.Linear(2, hidden_dim)
        self.feature_embedding = nn.Parameter(
            torch.empty(self.feature_count, hidden_dim)
        )
        self.node_embedding = nn.Parameter(torch.empty(self.node_count, hidden_dim))
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.node_embedding, std=0.02)
        self.register_buffer(
            "feature_node", torch.tensor(graph.feature_node_indices, dtype=torch.long)
        )
        self.register_buffer(
            "advective_feature",
            torch.tensor(
                [relation == "advective" for relation in graph.feature_relations],
                dtype=torch.bool,
            ),
        )

    def forward(self, x: Tensor, residual: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape != residual.shape or x.ndim != 3:
            raise ValueError("feature and residual tensors must share [B,T,F]")
        encoded = self.value_projection(torch.stack([x, residual], dim=-1))
        encoded = encoded + self.feature_embedding.view(1, 1, self.feature_count, -1)
        batch, steps = x.shape[:2]

        def scatter(relation_mask: Tensor) -> Tensor:
            source = encoded * relation_mask.view(1, 1, -1, 1).to(encoded.dtype)
            target = torch.zeros(
                batch,
                steps,
                self.node_count,
                self.hidden_dim,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            target.index_add_(2, self.feature_node, source)
            counts = torch.zeros(
                self.node_count, dtype=encoded.dtype, device=encoded.device
            )
            counts.index_add_(0, self.feature_node, relation_mask.to(encoded.dtype))
            target = target / counts.clamp_min(1.0).view(1, 1, -1, 1)
            target = target + self.node_embedding.view(1, 1, self.node_count, -1)
            return target * valid_mask.unsqueeze(-1).unsqueeze(-1).to(target.dtype)

        return scatter(self.advective_feature), scatter(~self.advective_feature)


class NodeTemporalEncoder(nn.Module):
    """Shared causal GRU applied independently to every process node."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        batch, steps, nodes, hidden = values.shape
        sequence = values.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        encoded, _ = self.gru(sequence)
        encoded = encoded.reshape(batch, nodes, steps, hidden).permute(0, 2, 1, 3)
        return encoded * valid_mask.unsqueeze(-1).unsqueeze(-1).to(encoded.dtype)


class TransportDelayBlock(nn.Module):
    """Mode-gated dual-relation message passing with fractional V/Q delay."""

    def __init__(
        self,
        graph: ProcessGraph,
        hidden_dim: int,
        dropout: float,
        maximum_prior_correction_fraction: float,
    ) -> None:
        super().__init__()
        self.node_count = len(graph.nodes)
        self.edge_count = len(graph.edges)
        self.max_correction = maximum_prior_correction_fraction
        self.route_gate = RouteGate(graph)
        self.advective_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.control_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.advective_self = nn.Linear(hidden_dim, hidden_dim)
        self.control_self = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embedding = nn.Parameter(torch.empty(self.edge_count, hidden_dim))
        self.prior_delay_parameter = nn.Parameter(torch.zeros(self.edge_count))
        nn.init.normal_(self.edge_embedding, std=0.02)
        self.advective_norm = nn.LayerNorm(hidden_dim)
        self.control_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "source", torch.tensor([edge.source for edge in graph.edges], dtype=torch.long)
        )
        self.register_buffer(
            "destination",
            torch.tensor([edge.destination for edge in graph.edges], dtype=torch.long),
        )
        self.register_buffer(
            "advective_edge",
            torch.tensor([edge.advective for edge in graph.edges], dtype=torch.bool),
        )
        self.register_buffer(
            "one_step_edge",
            torch.tensor(
                [edge.delay_kind == "exact_one_step_return" for edge in graph.edges],
                dtype=torch.bool,
            ),
        )
        self.register_buffer(
            "learnable_prior_edge",
            torch.tensor(
                [
                    edge.delay_kind in {"nominal_pid_prior", "perfect_mix_inventory"}
                    for edge in graph.edges
                ],
                dtype=torch.bool,
            ),
        )

    def _delayed_sources(
        self,
        advective: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if dt_s.shape != advective.shape[:2]:
            raise ValueError("dt_s shape mismatch")
        if edge_volume_l.shape != (advective.shape[0], self.edge_count):
            raise ValueError("edge_volume_l shape mismatch")
        if not torch.isfinite(dt_s).all() or torch.any(dt_s <= 0.0):
            raise ValueError("dt_s must be finite and positive")
        if not torch.isfinite(edge_volume_l).all() or torch.any(edge_volume_l < 0.0):
            raise ValueError("edge volumes must be finite and non-negative")
        flow = controls_raw[..., CONTROL_FIELDS.index("measured_flow_l_h")]
        safe_flow = flow.clamp_min(1.0)
        correction = torch.ones(
            self.edge_count,
            dtype=advective.dtype,
            device=advective.device,
        )
        bounded = self.max_correction * torch.tanh(self.prior_delay_parameter)
        correction = correction + bounded * self.learnable_prior_edge.to(advective.dtype)
        delay_seconds = (
            edge_volume_l[:, None, :]
            * 3600.0
            / safe_flow.unsqueeze(-1)
            * correction.view(1, 1, -1)
        )
        delay_steps = delay_seconds / dt_s.unsqueeze(-1)
        delay_steps = torch.where(
            self.one_step_edge.view(1, 1, -1),
            torch.ones_like(delay_steps),
            delay_steps,
        )
        delayed: list[Tensor] = []
        valid: list[Tensor] = []
        for edge_index in range(self.edge_count):
            source_history = advective[:, :, self.source[edge_index], :]
            sampled, sampled_valid = fractional_delay(
                source_history, delay_steps[:, :, edge_index]
            )
            delayed.append(sampled)
            valid.append(sampled_valid)
        return torch.stack(delayed, dim=2), torch.stack(valid, dim=2), delay_seconds

    def forward(
        self,
        advective: Tensor,
        control: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gate = self.route_gate(controls_raw)
        delayed, delay_valid, delay_seconds = self._delayed_sources(
            advective, controls_raw, dt_s, edge_volume_l
        )
        adv_gate = (
            gate
            * delay_valid.to(gate.dtype)
            * self.advective_edge.view(1, 1, -1).to(gate.dtype)
        )
        control_source = control.index_select(2, self.source)
        edge_bias = self.edge_embedding.view(1, 1, self.edge_count, -1)
        adv_messages = (self.advective_message(delayed) + edge_bias) * adv_gate.unsqueeze(-1)
        control_messages = (
            self.control_message(control_source) + edge_bias
        ) * gate.unsqueeze(-1)

        def aggregate(messages: Tensor, weights: Tensor) -> Tensor:
            result = torch.zeros_like(advective)
            result.index_add_(2, self.destination, messages)
            degree = torch.zeros(
                advective.shape[0],
                advective.shape[1],
                self.node_count,
                dtype=messages.dtype,
                device=messages.device,
            )
            degree.index_add_(2, self.destination, weights)
            return result / degree.clamp_min(1.0).unsqueeze(-1)

        adv_aggregate = aggregate(adv_messages, adv_gate)
        control_aggregate = aggregate(control_messages, gate)
        advective_new = self.advective_norm(
            self.advective_self(advective) + self.dropout(adv_aggregate)
        )
        control_new = self.control_norm(
            self.control_self(control) + self.dropout(control_aggregate)
        )
        advective_new = F.gelu(advective_new)
        control_new = F.gelu(control_new)
        mask = valid_mask.unsqueeze(-1).unsqueeze(-1).to(advective.dtype)
        advective_new = advective_new * mask
        control_new = control_new * mask

        destination_state = advective.index_select(2, self.destination)
        edge_error = (delayed - destination_state).square().mean(dim=-1)
        transport_error = (edge_error * adv_gate).sum(dim=-1) / adv_gate.sum(
            dim=-1
        ).clamp_min(1.0)
        transport_error = transport_error * valid_mask.to(transport_error.dtype)
        return advective_new, control_new, transport_error, delay_seconds

    def delay_regularization(self) -> Tensor:
        selected = self.prior_delay_parameter[self.learnable_prior_edge]
        if selected.numel() == 0:
            return self.prior_delay_parameter.sum() * 0.0
        return torch.tanh(selected).square().mean()


class FlowTwinGuard(nn.Module):
    def __init__(
        self,
        graph: ProcessGraph,
        config: FlowTwinConfig,
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
            raise ValueError("sensor uncertainty vector has wrong width")
        if not torch.isfinite(sensor_uncertainty_normalized).all() or torch.any(
            sensor_uncertainty_normalized < 0.0
        ):
            raise ValueError("sensor uncertainty must be finite and non-negative")
        self.config = config
        self.observer = NominalObserver(
            config.feature_count, config.observer_hidden_dim
        )
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
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.node_attention = nn.Linear(config.hidden_dim, 1)
        self.class_head = nn.Linear(config.hidden_dim, config.class_count)
        self.anomaly_head = nn.Linear(config.hidden_dim, 2)
        self.location_head = nn.Linear(config.hidden_dim, config.location_count)
        self.mechanism_head = nn.Linear(config.hidden_dim, config.mechanism_count)
        self.register_buffer(
            "sensor_uncertainty_normalized",
            sensor_uncertainty_normalized.detach().clone().to(torch.float32),
        )

    def set_observer_trainable(self, trainable: bool) -> None:
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
        mean, log_variance, residual, observer_valid = self.observer_outputs(
            x, valid_mask
        )
        advective, control = self.projector(x, residual, valid_mask)
        advective = self.advective_temporal(advective, valid_mask)
        control = self.control_temporal(control, valid_mask)
        transport_errors: list[Tensor] = []
        delay_seconds: Tensor | None = None
        for block in self.graph_blocks:
            advective, control, transport_error, delay_seconds = block(
                advective,
                control,
                controls_raw,
                dt_s,
                edge_volume_l,
                valid_mask,
            )
            transport_errors.append(transport_error)
        node_embedding = self.fusion(torch.cat([advective, control], dim=-1))
        attention = self.node_attention(node_embedding).squeeze(-1)
        attention = torch.softmax(attention, dim=-1)
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
            "delay_seconds": delay_seconds
            if delay_seconds is not None
            else torch.zeros_like(edge_volume_l[:, None, :]),
        }
        if include_node_embeddings:
            result["node_embedding"] = node_embedding
            result["node_attention"] = attention
        return result

    def observer_nll(self, x: Tensor, output: Mapping[str, Tensor], valid_mask: Tensor) -> Tensor:
        variance = torch.exp(output["nominal_log_variance"]) + self.sensor_uncertainty_normalized.square().view(
            1, 1, -1
        )
        per_feature = 0.5 * (
            (x - output["nominal_mean"]).square() / variance.clamp_min(1e-6)
            + torch.log(variance.clamp_min(1e-6))
        )
        mask = valid_mask & output["observer_valid"]
        if not torch.any(mask):
            return per_feature.sum() * 0.0
        return per_feature[mask].mean()

    def delay_regularization(self) -> Tensor:
        return torch.stack([block.delay_regularization() for block in self.graph_blocks]).mean()


def focal_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    gamma: float = 2.0,
    class_weight: Tensor | None = None,
) -> Tensor:
    selected = mask & (target >= 0)
    if not torch.any(selected):
        return logits.sum() * 0.0
    logits_selected = logits[selected]
    target_selected = target[selected]
    ce = F.cross_entropy(
        logits_selected,
        target_selected,
        reduction="none",
        weight=class_weight,
    )
    probability = torch.softmax(logits_selected, dim=-1).gather(
        1, target_selected.unsqueeze(-1)
    ).squeeze(-1)
    return (((1.0 - probability).clamp_min(0.0) ** gamma) * ce).mean()


def counterfactual_contrastive_loss(
    embedding: Tensor,
    reference_embedding: Tensor,
    class_target: Tensor,
    valid_mask: Tensor,
    *,
    normal_index: int,
    margin: float = 1.0,
) -> Tensor:
    if embedding.shape != reference_embedding.shape:
        raise ValueError("counterfactual embeddings must have equal shape")
    distance = torch.sqrt(
        (embedding - reference_embedding).square().mean(dim=-1).clamp_min(1e-12)
    )
    positive = class_target != normal_index
    loss = torch.where(positive, F.relu(margin - distance).square(), distance.square())
    selected = valid_mask & (class_target >= 0)
    if not torch.any(selected):
        return loss.sum() * 0.0
    return loss[selected].mean()


__all__ = [
    "MODEL_VERSION",
    "FlowTwinConfig",
    "FlowTwinGuard",
    "NominalObserver",
    "RouteGate",
    "TransportDelayBlock",
    "counterfactual_contrastive_loss",
    "focal_cross_entropy",
    "fractional_delay",
]
