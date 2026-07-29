#!/usr/bin/env python3
"""Tests for the leakage-safe FlowTwin-Guard benchmark baselines."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import torch  # noqa: E402

from flowtwin_guard.baselines import (  # noqa: E402
    AVAILABLE_BASELINES,
    BaselineConfig,
    TwinResidualTCNBaseline,
    build_baseline,
)
from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph  # noqa: E402


class FlowTwinBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3456)
        self.graph = build_process_graph()
        self.batch = 2
        self.steps = 13
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

    def _config(self, model_name: str) -> BaselineConfig:
        return BaselineConfig(
            model_name=model_name,
            feature_count=len(self.graph.features),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            class_count=20,
            location_count=9,
            mechanism_count=8,
            hidden_dim=8,
            layers=1,
            dropout=0.0,
            kernel_size=3,
            attention_heads=2,
            observer_hidden_dim=8,
        )

    def _model(self, model_name: str) -> torch.nn.Module:
        uncertainty = torch.linspace(0.0, 0.1, len(self.graph.features))
        return build_baseline(self.graph, self._config(model_name), uncertainty)

    def test_config_is_json_round_trip_and_version_checked(self) -> None:
        config = self._config("causal_transformer")
        encoded = json.loads(json.dumps(config.to_dict()))
        self.assertEqual(BaselineConfig.from_dict(encoded), config)
        encoded["baseline_version"] = "99.0"
        with self.assertRaisesRegex(ValueError, "unsupported baseline version"):
            BaselineConfig.from_dict(encoded)

    def test_all_factories_share_shapes_are_finite_and_trainable(self) -> None:
        required = {
            "class_logits": (self.batch, self.steps, 20),
            "anomaly_logits": (self.batch, self.steps, 2),
            "location_logits": (self.batch, self.steps, 9),
            "mechanism_logits": (self.batch, self.steps, 8),
            "ood_energy": (self.batch, self.steps),
            "embedding": (self.batch, self.steps, 8),
        }
        for name in AVAILABLE_BASELINES:
            with self.subTest(name=name):
                model = self._model(name)
                output = model(
                    self.x,
                    self.controls,
                    self.dt_s,
                    self.volumes,
                    self.valid,
                )
                for key, shape in required.items():
                    self.assertEqual(tuple(output[key].shape), shape)
                    self.assertTrue(torch.isfinite(output[key]).all())
                output["class_logits"].square().mean().backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_every_model_is_causal_under_future_perturbation(self) -> None:
        cutoff = 7
        changed_x = self.x.clone()
        changed_x[:, cutoff:] += 10_000.0
        changed_controls = self.controls.clone()
        changed_controls[:, cutoff:, :] += 500.0
        changed_dt = self.dt_s.clone()
        changed_dt[:, cutoff:] = 17.0
        for name in AVAILABLE_BASELINES:
            with self.subTest(name=name):
                model = self._model(name).eval()
                original = model(
                    self.x,
                    self.controls,
                    self.dt_s,
                    self.volumes,
                    self.valid,
                )
                changed = model(
                    changed_x,
                    changed_controls,
                    changed_dt,
                    self.volumes,
                    self.valid,
                )
                for key in (
                    "class_logits",
                    "anomaly_logits",
                    "location_logits",
                    "mechanism_logits",
                    "ood_energy",
                    "embedding",
                ):
                    torch.testing.assert_close(
                        original[key][:, :cutoff],
                        changed[key][:, :cutoff],
                        rtol=0.0,
                        atol=1e-6,
                    )

    def test_twin_observer_uses_previous_observation_only(self) -> None:
        model = self._model("twin_residual_tcn").eval()
        self.assertIsInstance(model, TwinResidualTCNBaseline)
        original = model(
            self.x,
            self.controls,
            self.dt_s,
            self.volumes,
            self.valid,
        )
        changed_x = self.x.clone()
        changed_x[:, 5] += 1000.0
        changed = model(
            changed_x,
            self.controls,
            self.dt_s,
            self.volumes,
            self.valid,
        )
        torch.testing.assert_close(
            original["nominal_mean"][:, :6],
            changed["nominal_mean"][:, :6],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertFalse(bool(original["observer_valid"][:, 0].any()))
        nll = model.observer_nll(self.x, original, self.valid)
        self.assertTrue(torch.isfinite(nll))

    def test_invalid_padding_is_zero_and_contract_errors_fail_closed(self) -> None:
        model = self._model("tcn").eval()
        valid = self.valid.clone()
        valid[:, -3:] = False
        output = model(
            self.x,
            self.controls,
            self.dt_s,
            self.volumes,
            valid,
        )
        for key in (
            "class_logits",
            "anomaly_logits",
            "location_logits",
            "mechanism_logits",
            "ood_energy",
            "embedding",
        ):
            self.assertEqual(float(output[key][:, -3:].detach().abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            model(
                self.x,
                self.controls,
                self.dt_s,
                self.volumes,
                self.valid.to(torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
