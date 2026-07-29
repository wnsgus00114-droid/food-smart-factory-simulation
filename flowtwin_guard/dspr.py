"""Causal DSPR diagnostic adaptation for the FlowTwin-Guard benchmark.

This module is an independent *diagnostic adaptation* of Zhang et al.,
``DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time
Series Forecasting`` (arXiv:2604.07393v3).  It is deliberately not labelled a
reproduction: the paper defines a forecasting model, its author implementation
was not public in the cited version, and this benchmark requires a per-sample
six-output fault-diagnosis contract.

The implementation preserves the paper's central architectural hypotheses:

* a causal multi-scale statistical trend stream;
* an additive, learnably gated physics-residual stream;
* static physical-prior/learned-topology fusion;
* an input-conditioned dynamic graph with no self-loops;
* a channel-specific learned temporal receptive field; and
* static/dynamic gated fusion plus the paper's graph regularizers.

It never uses FlowTwin-Guard route gates or edge ``V/Q`` delays.  The common
``controls_raw``, ``dt_s`` and ``edge_volume_l`` arguments are validated but do
not enter the computation.  Consequently, flow can influence an adaptive
window only as an observed, standardized feature in ``x``, as it does in the
paper's historical-observation formulation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph import CONTROL_FIELDS, ProcessGraph


DSPR_DIAGNOSTIC_ADAPTATION_NAME = "dspr_diagnostic_adaptation"
DSPR_DIAGNOSTIC_ADAPTATION_VERSION = "0.1.0"
DSPR_SOURCE_PAPER = {
    "title": (
        "DSPR: Dual-Stream Physics-Residual Networks for Trustworthy "
        "Industrial Time Series Forecasting"
    ),
    "authors": ["Yeran Zhang", "Pengwei Yang", "Guoqing Wang", "Tianyu Li"],
    "arxiv_id": "2604.07393",
    "version": "v3",
    "url": "https://arxiv.org/abs/2604.07393v3",
    "accessed_utc_date": "2026-07-27",
}

# These records are serialized with every config.  Keeping the deviations in
# executable metadata prevents a benchmark manifest from silently calling this
# adaptation an author-code reproduction.
DSPR_DIAGNOSTIC_DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "task_and_output",
        "paper": "multi-horizon single-target forecasting",
        "adaptation": "causal per-sample multi-task fault diagnosis",
        "reason": "FlowTwin-Guard requires six diagnostic outputs at every step",
    },
    {
        "id": "trend_stream",
        "paper": "TimeMixer trend forecaster",
        "adaptation": "causal rolling multi-scale MLP trend encoder",
        "reason": (
            "the cited DSPR author implementation was not public and a "
            "bidirectional forecasting backbone would violate online causality"
        ),
    },
    {
        "id": "adaptive_window_training",
        "paper": "hard channel-specific mask defined by continuous tau",
        "adaptation": "differentiable logistic relaxation of the hard boundary",
        "reason": (
            "the paper does not specify an estimator for gradients through its "
            "hard comparison; the relaxation makes tau learnable end-to-end"
        ),
    },
    {
        "id": "dynamic_prior_guidance",
        "paper": "dynamic similarity graph plus a separate prior-fused static graph",
        "adaptation": "prior-fused static adjacency also biases dynamic graph logits",
        "reason": (
            "this makes the requested physics-guided dynamic graph explicit while "
            "retaining dense data-driven edges instead of hard route gates"
        ),
    },
    {
        "id": "node_definition",
        "paper": "one graph node per process variable",
        "adaptation": "allowlisted features are scattered onto P&ID asset nodes",
        "reason": "the benchmark's physical prior is an asset-level directed graph",
    },
    {
        "id": "transport_inputs",
        "paper": "adaptive lags inferred from historical variable embeddings",
        "adaptation": (
            "controls, irregular dt and edge volumes are contract-validated but "
            "unused; tau is measured in samples"
        ),
        "reason": "prevents reuse of FlowTwin route gates or explicit edge V/Q delays",
    },
    {
        "id": "additive_fusion",
        "paper": "trend forecast plus gated residual forecast",
        "adaptation": "trend representation plus gated residual representation",
        "reason": "the combined representation feeds the required diagnostic heads",
    },
    {
        "id": "residual_gate_initialization",
        "paper": (
            "equation 2 defines alpha=sigmoid(beta) and states beta starts at zero, "
            "while adjacent prose calls the resulting gate zero"
        ),
        "adaptation": "beta/logit starts at zero, so the mathematical initial gate is 0.5",
        "reason": "follows the published equation rather than silently resolving its ambiguity",
    },
    {
        "id": "implementation_status",
        "paper": "implementation promised after manuscript acceptance",
        "adaptation": "independent implementation from equations and Algorithm 1",
        "reason": "no author implementation was linked by arXiv v3 when inspected",
    },
)


def adaptation_metadata() -> dict[str, Any]:
    """Return JSON-serializable provenance and scientific claim boundaries."""

    return {
        "adaptation_name": DSPR_DIAGNOSTIC_ADAPTATION_NAME,
        "adaptation_version": DSPR_DIAGNOSTIC_ADAPTATION_VERSION,
        "status": "closest_prior_art_diagnostic_adaptation_not_author_reproduction",
        "source_paper": {
            key: list(value) if isinstance(value, list) else value
            for key, value in DSPR_SOURCE_PAPER.items()
        },
        "implemented_components": [
            "dual additive trend/residual streams (paper equations 2 and 14)",
            "physical-prior and learned static topology fusion (equations 4-6)",
            "input-conditioned no-self-loop dynamic graph (equations 7 and 10)",
            "channel-specific adaptive causal temporal attention (equations 8-11)",
            "static/dynamic gated fusion (equations 12-13)",
            "physical-alignment and sparsity regularization (equation 15 and appendix A.3)",
        ],
        "deviations": [dict(item) for item in DSPR_DIAGNOSTIC_DEVIATIONS],
        "forbidden_flowtwin_mechanisms": [
            "route gates",
            "edge V/Q transport delays",
            "oracle plant mode",
            "fault modifiers or canonical labels as inputs",
        ],
        "scientific_limits": [
            (
                "The P&ID adjacency is a sparse plausible-interaction hypothesis "
                "space, not a ground-truth causal graph."
            ),
            (
                "Adaptive-window values are learned effective receptive fields in "
                "samples, not sensor-calibrated physical transport-time estimates."
            ),
            (
                "Diagnostic performance on synthetic D1 data cannot establish "
                "field accuracy, process safety, HACCP conformity or product release."
            ),
        ],
        "claim_boundary": (
            "This is a diagnostic adaptation used as prior-art comparison, not an "
            "exact reproduction of the paper's forecasting results."
        ),
    }


@dataclass(frozen=True)
class DSPRDiagnosticConfig:
    """Serializable configuration for the DSPR diagnostic adaptation."""

    feature_count: int
    node_count: int
    edge_count: int
    class_count: int
    location_count: int
    mechanism_count: int
    hidden_dim: int = 64
    trend_depth: int = 4
    trend_downsample_ratio: int = 2
    trend_kernel_size: int = 25
    attention_heads: int = 4
    max_adaptive_window: int = 20
    adaptive_window_temperature: float = 0.75
    prior_mix_initial: float = 0.5
    dynamic_prior_strength: float = 1.0
    residual_gate_initial_logit: float = 0.0
    dropout: float = 0.10
    physics_alignment_weight: float = 1.0e-2
    sparsity_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        for field in (
            "feature_count",
            "node_count",
            "edge_count",
            "class_count",
            "location_count",
            "mechanism_count",
            "hidden_dim",
            "trend_depth",
            "trend_downsample_ratio",
            "trend_kernel_size",
            "attention_heads",
            "max_adaptive_window",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.node_count < 2:
            raise ValueError("node_count must be at least two because self-loops are masked")
        if self.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for paper-style D/2 branches")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.adaptive_window_temperature) or (
            self.adaptive_window_temperature <= 0.0
        ):
            raise ValueError("adaptive_window_temperature must be finite and positive")
        if not 0.0 < self.prior_mix_initial < 1.0:
            raise ValueError("prior_mix_initial must be strictly between zero and one")
        for field in (
            "dynamic_prior_strength",
            "physics_alignment_weight",
            "sparsity_weight",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if not math.isfinite(self.residual_gate_initial_logit):
            raise ValueError("residual_gate_initial_logit must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {**adaptation_metadata(), "config": asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DSPRDiagnosticConfig":
        payload = dict(value)
        expected = adaptation_metadata()
        for field in (
            "adaptation_name",
            "adaptation_version",
            "status",
            "source_paper",
            "implemented_components",
            "deviations",
            "forbidden_flowtwin_mechanisms",
            "scientific_limits",
            "claim_boundary",
        ):
            if payload.get(field) != expected[field]:
                raise ValueError(f"DSPR adaptation metadata mismatch for {field}")
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("DSPR serialized config is missing its config object")
        return cls(**dict(config))


def _validate_inputs(
    config: DSPRDiagnosticConfig,
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
            f"[{batch},{steps},{len(CONTROL_FIELDS)}]"
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
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("every sequence must contain at least one valid sample")


class _DiagnosticHeads(nn.Module):
    """The exact six-output diagnostic contract used by matched baselines."""

    def __init__(self, config: DSPRDiagnosticConfig) -> None:
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
        ood_energy = -torch.logsumexp(class_logits, dim=-1)
        ood_energy = ood_energy * valid_mask.to(ood_energy.dtype)
        return {
            "class_logits": class_logits,
            "anomaly_logits": anomaly_logits,
            "location_logits": location_logits,
            "mechanism_logits": mechanism_logits,
            "ood_energy": ood_energy,
            "embedding": embedding,
        }


def _causal_masked_average(
    values: Tensor, valid_mask: Tensor, window: int
) -> Tensor:
    """Rolling mean over ``[t-window+1, t]`` with no future samples."""

    transposed = (values * valid_mask.unsqueeze(-1).to(values.dtype)).transpose(1, 2)
    padded = F.pad(transposed, (window - 1, 0))
    numerator = F.avg_pool1d(padded, kernel_size=window, stride=1) * window
    weights = valid_mask.to(values.dtype).unsqueeze(1)
    denominator = (
        F.avg_pool1d(F.pad(weights, (window - 1, 0)), window, stride=1) * window
    )
    result = (numerator / denominator.clamp_min(1.0)).transpose(1, 2)
    return result * valid_mask.unsqueeze(-1).to(result.dtype)


class _CausalMultiScaleMixerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        windows: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.windows = windows
        self.scale_mixer = nn.Sequential(
            nn.Linear(hidden_dim * len(windows), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        scales = [
            _causal_masked_average(values, valid_mask, window)
            for window in self.windows
        ]
        mixed = self.scale_mixer(torch.cat(scales, dim=-1))
        output = self.norm(values + mixed)
        return output * valid_mask.unsqueeze(-1).to(output.dtype)


class _CausalTrendStream(nn.Module):
    """Causal TimeMixer-style multi-scale trend adaptation."""

    def __init__(self, config: DSPRDiagnosticConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.feature_count, config.hidden_dim)
        # TimeMixer's kernel/downsampling hyperparameters are represented as
        # progressively wider one-sided supports.  Unlike interpolation after
        # downsampling, this construction cannot leak a future sample.
        self.windows = tuple(
            config.trend_kernel_size * config.trend_downsample_ratio**level
            for level in range(config.trend_depth)
        )
        self.blocks = nn.ModuleList(
            [
                _CausalMultiScaleMixerBlock(
                    config.hidden_dim, self.windows, config.dropout
                )
                for _ in range(config.trend_depth)
            ]
        )

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        mask = valid_mask.unsqueeze(-1).to(x.dtype)
        hidden = self.input_projection(x * mask) * mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        return hidden


class _FeatureNodeProjector(nn.Module):
    """Map allowlisted feature channels to asset nodes without oracle fields."""

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
        feature_node = torch.tensor(graph.feature_node_indices, dtype=torch.long)
        self.register_buffer("feature_node", feature_node)
        counts = torch.bincount(feature_node, minlength=self.node_count).clamp_min(1)
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
        return nodes * valid_mask[:, :, None, None].to(nodes.dtype)


class _PhysicsGuidedGraph(nn.Module):
    """Paper-style static prior fusion and regime-dependent graph construction."""

    def __init__(self, graph: ProcessGraph, config: DSPRDiagnosticConfig) -> None:
        super().__init__()
        self.node_count = config.node_count
        self.hidden_dim = config.hidden_dim
        self.dynamic_prior_strength = config.dynamic_prior_strength
        self.node_topology_embedding = nn.Parameter(
            torch.empty(config.node_count, config.hidden_dim)
        )
        nn.init.normal_(self.node_topology_embedding, std=0.02)
        prior_logit = math.log(
            config.prior_mix_initial / (1.0 - config.prior_mix_initial)
        )
        self.prior_mix_logit = nn.Parameter(torch.tensor(prior_logit))

        # A[destination, source] multiplies H[source] to aggregate physical
        # influence into a destination.  Parallel P&ID relations remain a
        # binary plausible-interaction prior rather than multiplicity weights.
        prior = torch.zeros(config.node_count, config.node_count)
        for edge in graph.edges:
            if edge.source != edge.destination:
                prior[edge.destination, edge.source] = 1.0
        self.register_buffer("physical_prior", prior)
        self.register_buffer(
            "diagonal_mask", torch.eye(config.node_count, dtype=torch.bool)
        )

    def learned_static_adjacency(self) -> Tensor:
        logits = F.relu(
            self.node_topology_embedding @ self.node_topology_embedding.transpose(0, 1)
        )
        logits = logits.masked_fill(self.diagonal_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def static_adjacency(self) -> Tensor:
        learned = self.learned_static_adjacency()
        row_sum = self.physical_prior.sum(dim=-1, keepdim=True)
        normalized_prior = self.physical_prior / row_sum.clamp_min(1.0)
        # Boundary/source nodes without a known incoming edge retain a learned
        # row; otherwise convex fusion would produce a non-stochastic matrix.
        normalized_prior = torch.where(row_sum > 0.0, normalized_prior, learned)
        mixing = torch.sigmoid(self.prior_mix_logit)
        adjacency = mixing * normalized_prior + (1.0 - mixing) * learned
        adjacency = adjacency.masked_fill(self.diagonal_mask, 0.0)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    def dynamic_adjacency(self, nodes: Tensor, static_adjacency: Tensor) -> Tensor:
        similarity = torch.matmul(nodes, nodes.transpose(-1, -2)) / math.sqrt(
            self.hidden_dim
        )
        if self.dynamic_prior_strength > 0.0:
            # A soft log-prior guides rather than hard-masks the time-varying
            # graph.  Thus previously unknown secondary couplings remain
            # learnable, matching the paper's stated limitation boundary.
            log_prior = torch.log(static_adjacency.clamp_min(1.0e-8))
            similarity = similarity + self.dynamic_prior_strength * log_prior
        similarity = similarity.masked_fill(
            self.diagonal_mask.view(1, 1, self.node_count, self.node_count),
            torch.finfo(similarity.dtype).min,
        )
        return torch.softmax(similarity, dim=-1)


class _AdaptiveCausalTemporalAttention(nn.Module):
    """Channel-specific, learned causal receptive fields (paper eqs. 8-11)."""

    def __init__(self, config: DSPRDiagnosticConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_count = config.attention_heads
        self.head_dim = config.hidden_dim // config.attention_heads
        self.max_window = float(config.max_adaptive_window)
        self.temperature = config.adaptive_window_temperature
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.window_projection = nn.Linear(config.hidden_dim, 1)
        self.dropout = nn.Dropout(config.dropout)

    def predict_window(self, nodes: Tensor) -> Tensor:
        return 1.0 + (self.max_window - 1.0) * torch.sigmoid(
            self.window_projection(nodes).squeeze(-1)
        )

    def forward(self, nodes: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, node_count, _ = nodes.shape

        def heads(projection: nn.Linear) -> Tensor:
            value = projection(nodes).permute(0, 2, 1, 3)
            return value.reshape(
                batch, node_count, steps, self.head_count, self.head_dim
            ).permute(0, 1, 3, 2, 4)

        query = heads(self.query)
        key = heads(self.key)
        value = heads(self.value)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        adaptive_window = self.predict_window(nodes).permute(0, 2, 1)
        time = torch.arange(steps, dtype=nodes.dtype, device=nodes.device)
        age = time[:, None] - time[None, :]  # query t minus key k
        soft_boundary = F.logsigmoid(
            (adaptive_window.unsqueeze(-1) - age.view(1, 1, steps, steps))
            / self.temperature
        )
        score = score + soft_boundary.unsqueeze(2)

        prohibited = age < 0.0
        prohibited = prohibited.view(1, 1, 1, steps, steps)
        prohibited = prohibited | (~valid_mask[:, None, None, None, :])
        score = score.masked_fill(prohibited, torch.finfo(score.dtype).min)
        attention = self.dropout(torch.softmax(score, dim=-1))
        temporal = torch.matmul(attention, value)
        temporal = temporal.permute(0, 1, 3, 2, 4).reshape(
            batch, node_count, steps, self.hidden_dim
        )
        temporal = temporal.permute(0, 2, 1, 3)
        temporal = self.output(temporal)
        temporal = temporal * valid_mask[:, :, None, None].to(temporal.dtype)
        adaptive_window = adaptive_window.permute(0, 2, 1)
        adaptive_window = adaptive_window * valid_mask[:, :, None].to(
            adaptive_window.dtype
        )
        return temporal, adaptive_window


class _PhysicsResidualStream(nn.Module):
    def __init__(self, graph: ProcessGraph, config: DSPRDiagnosticConfig) -> None:
        super().__init__()
        half = config.hidden_dim // 2
        self.projector = _FeatureNodeProjector(graph, config.hidden_dim)
        self.graph = _PhysicsGuidedGraph(graph, config)
        self.static_projection = nn.Linear(config.hidden_dim, half)
        self.dynamic_spatial_projection = nn.Linear(config.hidden_dim, half)
        self.temporal_attention = _AdaptiveCausalTemporalAttention(config)
        self.dynamic_temporal_projection = nn.Linear(config.hidden_dim, half)
        self.dynamic_gate = nn.Linear(config.hidden_dim, half)
        self.residual_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.node_pool_score = nn.Linear(config.hidden_dim, 1)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        nodes = self.projector(x, valid_mask)
        static_adjacency = self.graph.static_adjacency()

        static_aggregate = torch.einsum("ij,btjd->btid", static_adjacency, nodes)
        static_context = self.static_projection(static_aggregate)

        dynamic_adjacency = self.graph.dynamic_adjacency(nodes, static_adjacency)
        dynamic_aggregate = torch.matmul(dynamic_adjacency, nodes)
        spatial = F.relu(self.dynamic_spatial_projection(dynamic_aggregate))
        temporal, _adaptive_window = self.temporal_attention(nodes, valid_mask)
        temporal = self.dynamic_temporal_projection(temporal)
        gate = torch.sigmoid(self.dynamic_gate(nodes))
        dynamic_context = gate * spatial + (1.0 - gate) * temporal

        residual_nodes = self.residual_projection(
            torch.cat([static_context, dynamic_context], dim=-1)
        )
        residual_nodes = self.norm(nodes + self.dropout(residual_nodes))
        pool = torch.softmax(self.node_pool_score(residual_nodes).squeeze(-1), dim=-1)
        residual = (pool.unsqueeze(-1) * residual_nodes).sum(dim=2)
        return residual * valid_mask.unsqueeze(-1).to(residual.dtype)


class DSPRDiagnosticAdaptation(nn.Module):
    """Closest-prior DSPR adaptation under the benchmark diagnostic contract."""

    def __init__(self, graph: ProcessGraph, config: DSPRDiagnosticConfig) -> None:
        super().__init__()
        if (
            config.feature_count != len(graph.features)
            or config.node_count != len(graph.nodes)
            or config.edge_count != len(graph.edges)
        ):
            raise ValueError("DSPR config dimensions do not match process graph")
        self.config = config
        self.trend_stream = _CausalTrendStream(config)
        self.residual_stream = _PhysicsResidualStream(graph, config)
        self.residual_gate_logits = nn.Parameter(
            torch.full((config.hidden_dim,), config.residual_gate_initial_logit)
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
        trend = self.trend_stream(x, valid_mask)
        residual = self.residual_stream(x, valid_mask)
        gate = torch.sigmoid(self.residual_gate_logits).view(1, 1, -1)
        embedding = self.output_norm(trend + gate * residual)
        embedding = embedding * valid_mask.unsqueeze(-1).to(embedding.dtype)
        return self.heads(embedding, valid_mask)

    def physics_alignment_loss(self) -> Tensor:
        """Equation-15-style alignment on confirmed directed prior edges."""

        graph = self.residual_stream.graph
        adjacency = graph.static_adjacency()
        confirmed = graph.physical_prior > 0.0
        if not torch.any(confirmed):
            return adjacency.sum() * 0.0
        return (adjacency[confirmed] - graph.physical_prior[confirmed]).square().mean()

    def sparsity_loss(self) -> Tensor:
        """Penalize learned mass outside the stated physical hypothesis edges."""

        graph = self.residual_stream.graph
        adjacency = graph.static_adjacency()
        off_prior = (~graph.diagonal_mask) & (graph.physical_prior <= 0.0)
        if not torch.any(off_prior):
            return adjacency.sum() * 0.0
        return adjacency[off_prior].abs().mean()

    def auxiliary_loss(self) -> Tensor:
        """Return the registered DSPR graph regularization contribution."""

        return (
            self.config.physics_alignment_weight * self.physics_alignment_loss()
            + self.config.sparsity_weight * self.sparsity_loss()
        )

    def adaptation_metadata(self) -> dict[str, Any]:
        return adaptation_metadata()


def build_dspr_diagnostic_adaptation(
    graph: ProcessGraph, config: DSPRDiagnosticConfig
) -> DSPRDiagnosticAdaptation:
    """Construct the versioned diagnostic adaptation."""

    return DSPRDiagnosticAdaptation(graph, config)


# Readable aliases for benchmark registries that use "baseline" terminology.
DSPRDiagnosticBaseline = DSPRDiagnosticAdaptation
create_dspr_diagnostic_adaptation = build_dspr_diagnostic_adaptation


__all__ = [
    "DSPR_DIAGNOSTIC_ADAPTATION_NAME",
    "DSPR_DIAGNOSTIC_ADAPTATION_VERSION",
    "DSPR_DIAGNOSTIC_DEVIATIONS",
    "DSPR_SOURCE_PAPER",
    "DSPRDiagnosticAdaptation",
    "DSPRDiagnosticBaseline",
    "DSPRDiagnosticConfig",
    "adaptation_metadata",
    "build_dspr_diagnostic_adaptation",
    "create_dspr_diagnostic_adaptation",
]
