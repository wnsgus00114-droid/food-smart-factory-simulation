"""Leakage-safe PyTorch baselines for the FlowTwin-Guard benchmark.

The models in this module intentionally share one deployable input contract and
one diagnostic output contract.  They differ only in the representation under
test:

* :class:`TCNBaseline` is a causal dilated-convolution sequence model;
* :class:`CausalTransformerBaseline` uses masked self-attention;
* :class:`StaticPIDGNNBaseline` uses the fixed directed P&ID graph;
* :class:`LearnedDynamicGNNBaseline` infers a dense graph at every sample; and
* :class:`TwinResidualTCNBaseline` adds a one-step nominal observer.

No baseline receives simulator truth, fault modifiers, PLC alarms, HACCP
outcomes, or a future observation.  ``controls_raw`` and the graph volume
arguments remain in every forward signature so a benchmark runner cannot
silently give one model a different input contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph import CONTROL_FIELDS, ProcessGraph


BASELINE_VERSION = "0.1.0"
AVAILABLE_BASELINES = (
    "tcn",
    "causal_transformer",
    "static_pid_gnn",
    "dynamic_gnn",
    "twin_residual_tcn",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Serializable configuration shared by all neural baselines."""

    model_name: str
    feature_count: int
    node_count: int
    edge_count: int
    class_count: int
    location_count: int
    mechanism_count: int
    hidden_dim: int = 32
    layers: int = 2
    dropout: float = 0.10
    kernel_size: int = 3
    attention_heads: int = 4
    observer_hidden_dim: int = 48

    def __post_init__(self) -> None:
        if self.model_name not in AVAILABLE_BASELINES:
            raise ValueError(
                f"unknown baseline {self.model_name!r}; expected one of "
                f"{AVAILABLE_BASELINES}"
            )
        for field in (
            "feature_count",
            "node_count",
            "edge_count",
            "class_count",
            "location_count",
            "mechanism_count",
            "hidden_dim",
            "layers",
            "kernel_size",
            "attention_heads",
            "observer_hidden_dim",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.model_name == "causal_transformer" and (
            self.hidden_dim % self.attention_heads != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible by attention_heads for the transformer"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, versioned representation."""

        return {"baseline_version": BASELINE_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaselineConfig":
        data = dict(value)
        version = data.pop("baseline_version", BASELINE_VERSION)
        if version != BASELINE_VERSION:
            raise ValueError(f"unsupported baseline version {version!r}")
        return cls(**data)


def _validate_inputs(
    config: BaselineConfig,
    x: Tensor,
    controls_raw: Tensor,
    dt_s: Tensor,
    edge_volume_l: Tensor,
    valid_mask: Tensor,
) -> None:
    """Enforce the common five-tensor benchmark contract."""

    if x.ndim != 3 or x.shape[-1] != config.feature_count:
        raise ValueError(
            f"x must have shape [B,T,{config.feature_count}], got {tuple(x.shape)}"
        )
    batch, steps = x.shape[:2]
    expected_controls = (batch, steps, len(CONTROL_FIELDS))
    if controls_raw.shape != expected_controls:
        raise ValueError(
            f"controls_raw must have shape {expected_controls}, got "
            f"{tuple(controls_raw.shape)}"
        )
    if dt_s.shape != (batch, steps):
        raise ValueError("dt_s must have shape [B,T]")
    if edge_volume_l.shape != (batch, config.edge_count):
        raise ValueError(
            f"edge_volume_l must have shape [B,{config.edge_count}]"
        )
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
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("every sequence must contain at least one valid sample")


class _DiagnosticHeads(nn.Module):
    """Identical heads keep output capacity fixed across baselines."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.class_head = nn.Linear(config.hidden_dim, config.class_count)
        self.anomaly_head = nn.Linear(config.hidden_dim, 2)
        self.location_head = nn.Linear(config.hidden_dim, config.location_count)
        self.mechanism_head = nn.Linear(config.hidden_dim, config.mechanism_count)

    def forward(self, embedding: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
        mask = valid_mask.unsqueeze(-1).to(embedding.dtype)
        embedding = embedding * mask
        class_logits = self.class_head(embedding) * mask
        anomaly_logits = self.anomaly_head(embedding) * mask
        location_logits = self.location_head(embedding) * mask
        mechanism_logits = self.mechanism_head(embedding) * mask
        energy = -torch.logsumexp(class_logits, dim=-1)
        energy = energy * valid_mask.to(energy.dtype)
        return {
            "class_logits": class_logits,
            "anomaly_logits": anomaly_logits,
            "location_logits": location_logits,
            "mechanism_logits": mechanism_logits,
            "ood_energy": energy,
            "embedding": embedding,
        }


class _CausalConv1d(nn.Module):
    """Left-padded convolution; no right/future padding is ever introduced."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, values: Tensor) -> Tensor:
        values = F.pad(values, (self.left_padding, 0))
        return self.convolution(values)


class _TemporalBlock(nn.Module):
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
        hidden = self.dropout(F.gelu(self.norm_2(hidden)))
        hidden = residual + hidden
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
                _TemporalBlock(
                    hidden_dim,
                    kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        mask = valid_mask.unsqueeze(-1).to(values.dtype)
        hidden = self.input_projection(values * mask) * mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        return self.output_norm(hidden) * mask


class TCNBaseline(nn.Module):
    """Causal TCN operating on the allowlisted standardized features."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        if config.model_name != "tcn":
            raise ValueError("TCNBaseline requires model_name='tcn'")
        self.config = config
        self.encoder = _CausalTCN(
            config.feature_count,
            config.hidden_dim,
            config.layers,
            config.kernel_size,
            config.dropout,
        )
        self.heads = _DiagnosticHeads(config)

    def forward(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        _validate_inputs(
            self.config, x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        return self.heads(self.encoder(x, valid_mask), valid_mask)


def _continuous_time_encoding(
    dt_s: Tensor,
    valid_mask: Tensor,
    hidden_dim: int,
    dtype: torch.dtype,
) -> Tensor:
    """Sinusoidal encoding of elapsed time using current and past ``dt_s`` only."""

    elapsed = torch.cumsum(dt_s * valid_mask.to(dt_s.dtype), dim=1) - dt_s
    elapsed = elapsed.clamp_min(0.0)
    pairs = (hidden_dim + 1) // 2
    exponent = torch.arange(pairs, device=dt_s.device, dtype=dt_s.dtype)
    denominator = max(1, pairs - 1)
    frequency = torch.exp(-math.log(10_000.0) * exponent / denominator)
    angles = elapsed.unsqueeze(-1) * frequency.view(1, 1, -1)
    encoded = torch.zeros(
        *dt_s.shape,
        hidden_dim,
        dtype=dt_s.dtype,
        device=dt_s.device,
    )
    encoded[..., 0::2] = torch.sin(angles[..., : encoded[..., 0::2].shape[-1]])
    encoded[..., 1::2] = torch.cos(angles[..., : encoded[..., 1::2].shape[-1]])
    return encoded.to(dtype) * valid_mask.unsqueeze(-1).to(dtype)


class CausalTransformerBaseline(nn.Module):
    """Transformer encoder with an explicit upper-triangular attention mask."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        if config.model_name != "causal_transformer":
            raise ValueError(
                "CausalTransformerBaseline requires model_name='causal_transformer'"
            )
        self.config = config
        self.input_projection = nn.Linear(config.feature_count, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            # Explicit dense tensors make the causal and padding masks behave
            # identically across CPU, CUDA and MPS benchmark runs.
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.heads = _DiagnosticHeads(config)

    def forward(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        _validate_inputs(
            self.config, x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        mask = valid_mask.unsqueeze(-1).to(x.dtype)
        hidden = self.input_projection(x * mask)
        hidden = hidden + _continuous_time_encoding(
            dt_s, valid_mask, self.config.hidden_dim, hidden.dtype
        )
        steps = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~valid_mask,
        )
        hidden = self.output_norm(hidden) * mask
        return self.heads(hidden, valid_mask)


class _FeatureNodeProjector(nn.Module):
    """Scatter scalar observations onto the assets defined by the P&ID adapter."""

    def __init__(self, graph: ProcessGraph, hidden_dim: int) -> None:
        super().__init__()
        self.feature_count = len(graph.features)
        self.node_count = len(graph.nodes)
        self.hidden_dim = hidden_dim
        self.value_projection = nn.Linear(1, hidden_dim)
        self.feature_embedding = nn.Parameter(
            torch.empty(self.feature_count, hidden_dim)
        )
        self.node_embedding = nn.Parameter(torch.empty(self.node_count, hidden_dim))
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.node_embedding, std=0.02)
        self.register_buffer(
            "feature_node",
            torch.tensor(graph.feature_node_indices, dtype=torch.long),
        )
        counts = torch.bincount(
            self.feature_node, minlength=self.node_count
        ).clamp_min(1)
        self.register_buffer("feature_count_by_node", counts.to(torch.float32))

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        encoded = self.value_projection(x.unsqueeze(-1))
        encoded = encoded + self.feature_embedding.view(
            1, 1, self.feature_count, self.hidden_dim
        )
        batch, steps = x.shape[:2]
        nodes = torch.zeros(
            batch,
            steps,
            self.node_count,
            self.hidden_dim,
            dtype=x.dtype,
            device=x.device,
        )
        nodes.index_add_(2, self.feature_node, encoded)
        nodes = nodes / self.feature_count_by_node.to(x.dtype).view(1, 1, -1, 1)
        nodes = nodes + self.node_embedding.view(1, 1, self.node_count, -1)
        return nodes * valid_mask.unsqueeze(-1).unsqueeze(-1).to(x.dtype)


class _StaticGraphLayer(nn.Module):
    def __init__(self, graph: ProcessGraph, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_count = len(graph.nodes)
        self.edge_count = len(graph.edges)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embedding = nn.Parameter(torch.empty(self.edge_count, hidden_dim))
        nn.init.normal_(self.edge_embedding, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "source", torch.tensor([edge.source for edge in graph.edges], dtype=torch.long)
        )
        self.register_buffer(
            "destination",
            torch.tensor([edge.destination for edge in graph.edges], dtype=torch.long),
        )
        degree = torch.bincount(self.destination, minlength=self.node_count).clamp_min(1)
        self.register_buffer("in_degree", degree.to(torch.float32))

    def forward(self, nodes: Tensor, valid_mask: Tensor) -> Tensor:
        source = nodes.index_select(2, self.source)
        messages = self.message(source) + self.edge_embedding.view(
            1, 1, self.edge_count, -1
        )
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(2, self.destination, messages)
        aggregate = aggregate / self.in_degree.to(nodes.dtype).view(1, 1, -1, 1)
        hidden = self.self_projection(nodes) + self.dropout(aggregate)
        hidden = F.gelu(self.norm(hidden))
        return hidden * valid_mask.unsqueeze(-1).unsqueeze(-1).to(nodes.dtype)


class _DynamicGraphLayer(nn.Module):
    """Input-conditioned dense graph attention with no P&ID adjacency prior."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, valid_mask: Tensor) -> Tensor:
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            nodes.shape[-1]
        )
        adjacency = torch.softmax(score, dim=-1)
        aggregate = torch.matmul(adjacency, value)
        hidden = nodes + self.dropout(self.output(aggregate))
        hidden = F.gelu(self.norm(hidden))
        return hidden * valid_mask.unsqueeze(-1).unsqueeze(-1).to(nodes.dtype)


class _GraphDiagnosticBaseline(nn.Module):
    def __init__(self, graph: ProcessGraph, config: BaselineConfig, *, dynamic: bool) -> None:
        super().__init__()
        self.config = config
        self.projector = _FeatureNodeProjector(graph, config.hidden_dim)
        if dynamic:
            self.graph_layers = nn.ModuleList(
                [
                    _DynamicGraphLayer(config.hidden_dim, config.dropout)
                    for _ in range(config.layers)
                ]
            )
        else:
            self.graph_layers = nn.ModuleList(
                [
                    _StaticGraphLayer(graph, config.hidden_dim, config.dropout)
                    for _ in range(config.layers)
                ]
            )
        self.node_attention = nn.Linear(config.hidden_dim, 1)
        self.temporal = _CausalTCN(
            config.hidden_dim,
            config.hidden_dim,
            config.layers,
            config.kernel_size,
            config.dropout,
        )
        self.heads = _DiagnosticHeads(config)

    def forward(
        self,
        x: Tensor,
        controls_raw: Tensor,
        dt_s: Tensor,
        edge_volume_l: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        _validate_inputs(
            self.config, x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        nodes = self.projector(x, valid_mask)
        for layer in self.graph_layers:
            nodes = layer(nodes, valid_mask)
        attention = torch.softmax(self.node_attention(nodes).squeeze(-1), dim=-1)
        pooled = (attention.unsqueeze(-1) * nodes).sum(dim=2)
        embedding = self.temporal(pooled, valid_mask)
        return self.heads(embedding, valid_mask)


class StaticPIDGNNBaseline(_GraphDiagnosticBaseline):
    """Directed static P&ID-GNN without route gates or transport delays."""

    def __init__(self, graph: ProcessGraph, config: BaselineConfig) -> None:
        if config.model_name != "static_pid_gnn":
            raise ValueError(
                "StaticPIDGNNBaseline requires model_name='static_pid_gnn'"
            )
        super().__init__(graph, config, dynamic=False)


class LearnedDynamicGNNBaseline(_GraphDiagnosticBaseline):
    """Input-conditioned learned graph without a fixed P&ID topology."""

    def __init__(self, graph: ProcessGraph, config: BaselineConfig) -> None:
        if config.model_name != "dynamic_gnn":
            raise ValueError(
                "LearnedDynamicGNNBaseline requires model_name='dynamic_gnn'"
            )
        super().__init__(graph, config, dynamic=True)


class _OneStepNominalObserver(nn.Module):
    """Predict the current healthy signal from observations through ``t-1``."""

    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.gru = nn.GRU(feature_count, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, feature_count)
        self.log_variance_head = nn.Linear(hidden_dim, feature_count)

    def forward(self, x: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        masked = x * valid_mask.unsqueeze(-1).to(x.dtype)
        shifted = torch.zeros_like(masked)
        shifted[:, 1:] = masked[:, :-1]
        hidden, _ = self.gru(shifted)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 4.0)
        observer_valid = valid_mask.clone()
        observer_valid[:, 0] = False
        observer_valid[:, 1:] &= valid_mask[:, :-1]
        mask = valid_mask.unsqueeze(-1).to(x.dtype)
        return mean * mask, log_variance * mask, observer_valid


class TwinResidualTCNBaseline(nn.Module):
    """TCN over observations and uncertainty-normalized nominal residuals."""

    def __init__(
        self,
        config: BaselineConfig,
        sensor_uncertainty_normalized: Tensor,
    ) -> None:
        super().__init__()
        if config.model_name != "twin_residual_tcn":
            raise ValueError(
                "TwinResidualTCNBaseline requires model_name='twin_residual_tcn'"
            )
        if sensor_uncertainty_normalized.shape != (config.feature_count,):
            raise ValueError("sensor uncertainty vector has the wrong width")
        if not torch.isfinite(sensor_uncertainty_normalized).all() or torch.any(
            sensor_uncertainty_normalized < 0.0
        ):
            raise ValueError("sensor uncertainty must be finite and non-negative")
        self.config = config
        self.observer = _OneStepNominalObserver(
            config.feature_count, config.observer_hidden_dim
        )
        self.encoder = _CausalTCN(
            config.feature_count * 2,
            config.hidden_dim,
            config.layers,
            config.kernel_size,
            config.dropout,
        )
        self.heads = _DiagnosticHeads(config)
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
    ) -> dict[str, Tensor]:
        _validate_inputs(
            self.config, x, controls_raw, dt_s, edge_volume_l, valid_mask
        )
        mean, log_variance, residual, observer_valid = self.observer_outputs(
            x, valid_mask
        )
        embedding = self.encoder(torch.cat([x, residual], dim=-1), valid_mask)
        result = self.heads(embedding, valid_mask)
        result.update(
            {
                "nominal_mean": mean,
                "nominal_log_variance": log_variance,
                "residual": residual,
                "observer_valid": observer_valid,
            }
        )
        return result

    def observer_nll(
        self, x: Tensor, output: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tensor:
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


def build_baseline(
    graph: ProcessGraph,
    config: BaselineConfig,
    sensor_uncertainty_normalized: Tensor | None = None,
) -> nn.Module:
    """Construct a configured baseline after checking graph compatibility."""

    if (
        config.feature_count != len(graph.features)
        or config.node_count != len(graph.nodes)
        or config.edge_count != len(graph.edges)
    ):
        raise ValueError("baseline config dimensions do not match process graph")
    if config.model_name == "tcn":
        return TCNBaseline(config)
    if config.model_name == "causal_transformer":
        return CausalTransformerBaseline(config)
    if config.model_name == "static_pid_gnn":
        return StaticPIDGNNBaseline(graph, config)
    if config.model_name == "dynamic_gnn":
        return LearnedDynamicGNNBaseline(graph, config)
    if sensor_uncertainty_normalized is None:
        raise ValueError("twin_residual_tcn requires sensor uncertainty")
    return TwinResidualTCNBaseline(config, sensor_uncertainty_normalized)


# A readable alias for callers that name constructors rather than builders.
create_baseline = build_baseline


__all__ = [
    "AVAILABLE_BASELINES",
    "BASELINE_VERSION",
    "BaselineConfig",
    "CausalTransformerBaseline",
    "LearnedDynamicGNNBaseline",
    "StaticPIDGNNBaseline",
    "TCNBaseline",
    "TwinResidualTCNBaseline",
    "build_baseline",
    "create_baseline",
]
