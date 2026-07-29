#!/usr/bin/env python3
"""Contract, causality and provenance tests for the DSPR adaptation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import torch  # noqa: E402

from flowtwin_guard.dspr import (  # noqa: E402
    DSPR_DIAGNOSTIC_ADAPTATION_NAME,
    DSPR_DIAGNOSTIC_ADAPTATION_VERSION,
    DSPRDiagnosticAdaptation,
    DSPRDiagnosticConfig,
    adaptation_metadata,
    build_dspr_diagnostic_adaptation,
)
from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph  # noqa: E402


class FlowTwinDSPRTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7319)
        self.graph = build_process_graph()
        self.batch = 2
        self.steps = 11
        self.x = torch.randn(self.batch, self.steps, len(self.graph.features))
        self.controls = torch.zeros(
            self.batch, self.steps, len(CONTROL_FIELDS), dtype=torch.float32
        )
        self.controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 20_000.0
        self.controls[..., CONTROL_FIELDS.index("power_good_signal")] = 1.0
        self.dt_s = torch.ones(self.batch, self.steps)
        profile = {
            "nominal_flow_l_h": 20_000.0,
            "nominal_holding_time_s": 18.0,
            "sensor_to_fdv_delay_s": 1.5,
            "post_fdv_residence_time_s": 5.0,
        }
        self.volumes = torch.tensor(
            self.graph.edge_volumes_for_profile(profile), dtype=torch.float32
        ).repeat(self.batch, 1)
        self.valid = torch.ones(self.batch, self.steps, dtype=torch.bool)
        self.config = DSPRDiagnosticConfig(
            feature_count=len(self.graph.features),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            class_count=20,
            location_count=9,
            mechanism_count=8,
            hidden_dim=8,
            trend_depth=2,
            trend_downsample_ratio=2,
            trend_kernel_size=3,
            attention_heads=2,
            max_adaptive_window=6,
            adaptive_window_temperature=0.75,
            dropout=0.0,
        )

    def _model(self) -> DSPRDiagnosticAdaptation:
        return build_dspr_diagnostic_adaptation(self.graph, self.config)

    def _forward(
        self,
        model: DSPRDiagnosticAdaptation,
        *,
        x: torch.Tensor | None = None,
        controls: torch.Tensor | None = None,
        dt_s: torch.Tensor | None = None,
        volumes: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return model(
            self.x if x is None else x,
            self.controls if controls is None else controls,
            self.dt_s if dt_s is None else dt_s,
            self.volumes if volumes is None else volumes,
            self.valid if valid is None else valid,
        )

    def test_config_round_trip_forces_adaptation_status_and_deviations(self) -> None:
        encoded = json.loads(json.dumps(self.config.to_dict()))
        self.assertEqual(encoded["adaptation_name"], DSPR_DIAGNOSTIC_ADAPTATION_NAME)
        self.assertEqual(
            encoded["adaptation_version"], DSPR_DIAGNOSTIC_ADAPTATION_VERSION
        )
        deviation_ids = {item["id"] for item in encoded["deviations"]}
        self.assertTrue(
            {
                "task_and_output",
                "trend_stream",
                "adaptive_window_training",
                "dynamic_prior_guidance",
                "transport_inputs",
                "implementation_status",
            }.issubset(deviation_ids)
        )
        self.assertTrue(
            any("not a ground-truth causal graph" in item for item in encoded["scientific_limits"])
        )
        self.assertEqual(DSPRDiagnosticConfig.from_dict(encoded), self.config)
        encoded["status"] = "exact_reproduction"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            DSPRDiagnosticConfig.from_dict(encoded)

    def test_six_output_contract_is_finite_trainable_and_adaptive(self) -> None:
        model = self._model()
        output = self._forward(model)
        expected = {
            "class_logits": (self.batch, self.steps, 20),
            "anomaly_logits": (self.batch, self.steps, 2),
            "location_logits": (self.batch, self.steps, 9),
            "mechanism_logits": (self.batch, self.steps, 8),
            "ood_energy": (self.batch, self.steps),
            "embedding": (self.batch, self.steps, 8),
        }
        self.assertEqual(set(output), set(expected))
        for key, shape in expected.items():
            self.assertEqual(tuple(output[key].shape), shape)
            self.assertTrue(torch.isfinite(output[key]).all())

        loss = (
            output["class_logits"].square().mean()
            + output["anomaly_logits"].square().mean()
            + model.auxiliary_loss()
        )
        loss.backward()
        window_gradient = (
            model.residual_stream.temporal_attention.window_projection.weight.grad
        )
        self.assertIsNotNone(window_gradient)
        self.assertTrue(torch.isfinite(window_gradient).all())
        self.assertGreater(float(window_gradient.abs().sum()), 0.0)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_future_perturbation_cannot_change_any_past_output(self) -> None:
        model = self._model().eval()
        cutoff = 6
        original = self._forward(model)
        changed_x = self.x.clone()
        changed_x[:, cutoff:] += 10_000.0
        changed_controls = self.controls.clone()
        changed_controls[:, cutoff:] += 500.0
        changed_dt = self.dt_s.clone()
        changed_dt[:, cutoff:] = 17.0
        changed_volumes = self.volumes * 23.0
        changed = self._forward(
            model,
            x=changed_x,
            controls=changed_controls,
            dt_s=changed_dt,
            volumes=changed_volumes,
        )
        for key in original:
            torch.testing.assert_close(
                original[key][:, :cutoff],
                changed[key][:, :cutoff],
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_route_controls_and_edge_v_over_q_are_not_computational_inputs(self) -> None:
        model = self._model().eval()
        original = self._forward(model)
        changed_controls = torch.randn_like(self.controls) * 1.0e5
        changed_dt = torch.full_like(self.dt_s, 91.0)
        changed_volumes = self.volumes * 1000.0 + 1.0
        changed = self._forward(
            model,
            controls=changed_controls,
            dt_s=changed_dt,
            volumes=changed_volumes,
        )
        for key in original:
            torch.testing.assert_close(
                original[key], changed[key], rtol=0.0, atol=0.0
            )
        metadata = adaptation_metadata()
        self.assertIn("route gates", metadata["forbidden_flowtwin_mechanisms"])
        self.assertIn(
            "edge V/Q transport delays", metadata["forbidden_flowtwin_mechanisms"]
        )

    def test_physical_graph_has_directed_prior_no_self_loops_and_regularizes(self) -> None:
        model = self._model()
        graph_module = model.residual_stream.graph
        self.assertGreater(float(graph_module.physical_prior.sum()), 0.0)
        self.assertEqual(
            float(torch.diagonal(graph_module.physical_prior).abs().sum()), 0.0
        )
        adjacency = graph_module.static_adjacency()
        self.assertTrue(torch.isfinite(adjacency).all())
        torch.testing.assert_close(
            adjacency.sum(dim=-1), torch.ones(len(self.graph.nodes)), atol=1.0e-6, rtol=0.0
        )
        self.assertEqual(
            float(torch.diagonal(adjacency).detach().abs().sum()), 0.0
        )
        self.assertTrue(torch.isfinite(model.physics_alignment_loss()))
        self.assertTrue(torch.isfinite(model.sparsity_loss()))
        self.assertTrue(torch.isfinite(model.auxiliary_loss()))

        nodes = model.residual_stream.projector(self.x, self.valid)
        dynamic = graph_module.dynamic_adjacency(nodes, adjacency)
        self.assertEqual(
            tuple(dynamic.shape),
            (self.batch, self.steps, len(self.graph.nodes), len(self.graph.nodes)),
        )
        self.assertTrue(torch.isfinite(dynamic).all())
        torch.testing.assert_close(
            dynamic.sum(dim=-1),
            torch.ones(self.batch, self.steps, len(self.graph.nodes)),
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertEqual(
            float(torch.diagonal(dynamic, dim1=-2, dim2=-1).detach().abs().sum()),
            0.0,
        )

        windows = model.residual_stream.temporal_attention.predict_window(nodes)
        self.assertGreaterEqual(float(windows.detach().min()), 1.0)
        self.assertLessEqual(
            float(windows.detach().max()), float(self.config.max_adaptive_window)
        )
        self.assertGreater(float(windows.detach().std()), 0.0)

    def test_padding_is_zero_and_contract_fails_closed(self) -> None:
        model = self._model().eval()
        valid = self.valid.clone()
        valid[:, -3:] = False
        output = self._forward(model, valid=valid)
        for key in output:
            self.assertEqual(float(output[key][:, -3:].detach().abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self._forward(model, valid=self.valid.to(torch.float32))
        bad_dt = self.dt_s.clone()
        bad_dt[:, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            self._forward(model, dt_s=bad_dt)


if __name__ == "__main__":
    unittest.main()
