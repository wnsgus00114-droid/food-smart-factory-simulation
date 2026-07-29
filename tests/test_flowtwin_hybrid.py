#!/usr/bin/env python3
"""Contract and invariance tests for the causal FlowTwin--TCN hybrid."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import torch  # noqa: E402

from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph  # noqa: E402
from flowtwin_guard.hybrid import (  # noqa: E402
    HYBRID_MODEL_VERSION,
    FlowTwinHybrid,
    FlowTwinHybridConfig,
    PersistenceSkipObserver,
)


class FlowTwinHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7281)
        self.graph = build_process_graph()
        self.batch = 2
        self.steps = 48
        self.x = torch.randn(self.batch, self.steps, len(self.graph.features))
        self.controls = torch.zeros(
            self.batch, self.steps, len(CONTROL_FIELDS), dtype=torch.float32
        )
        self.controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 20_000.0
        self.controls[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 1.0
        self.controls[..., CONTROL_FIELDS.index("steam_valve")] = 0.6
        self.controls[..., CONTROL_FIELDS.index("power_good_signal")] = 1.0
        self.dt_s = torch.full((self.batch, self.steps), 0.5)
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
        self.config = FlowTwinHybridConfig(
            feature_count=len(self.graph.features),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            class_count=20,
            fault_class_indices=tuple(range(3, 20)),
            location_count=9,
            mechanism_count=8,
            hidden_dim=8,
            temporal_layers=1,
            graph_layers=1,
            kernel_size=3,
            dropout=0.0,
            observer_hidden_dim=8,
        )
        self.uncertainty = torch.linspace(0.0, 0.1, len(self.graph.features))

    def _model(self) -> FlowTwinHybrid:
        return FlowTwinHybrid(self.graph, self.config, self.uncertainty)

    def _forward(self, model: FlowTwinHybrid, **changes: torch.Tensor) -> dict[str, torch.Tensor]:
        return model(
            changes.get("x", self.x),
            changes.get("controls", self.controls),
            changes.get("dt_s", self.dt_s),
            changes.get("volumes", self.volumes),
            changes.get("valid", self.valid),
        )

    def test_config_has_strict_versioned_json_round_trip(self) -> None:
        encoded = json.loads(json.dumps(self.config.to_dict()))
        self.assertEqual(encoded["hybrid_model_version"], HYBRID_MODEL_VERSION)
        self.assertEqual(FlowTwinHybridConfig.from_dict(encoded), self.config)

        missing = dict(encoded)
        missing.pop("kernel_size")
        with self.assertRaisesRegex(ValueError, "missing hybrid configuration"):
            FlowTwinHybridConfig.from_dict(missing)

        unknown = dict(encoded)
        unknown["future_oracle"] = True
        with self.assertRaisesRegex(ValueError, "unknown hybrid configuration"):
            FlowTwinHybridConfig.from_dict(unknown)

        stale = dict(encoded)
        stale["hybrid_model_version"] = "99.0.0"
        with self.assertRaisesRegex(ValueError, "unsupported hybrid model version"):
            FlowTwinHybridConfig.from_dict(stale)

    def test_six_output_contract_is_finite_and_trainable(self) -> None:
        model = self._model()
        output = self._forward(model)
        required = {
            "class_logits": (self.batch, self.steps, 20),
            "anomaly_logits": (self.batch, self.steps, 2),
            "location_logits": (self.batch, self.steps, 9),
            "mechanism_logits": (self.batch, self.steps, 8),
            "embedding": (self.batch, self.steps, 8),
            "ood_energy": (self.batch, self.steps),
        }
        for name, shape in required.items():
            self.assertEqual(tuple(output[name].shape), shape)
            self.assertTrue(torch.isfinite(output[name]).all(), name)

        loss = (
            output["class_logits"].square().mean()
            + output["anomaly_logits"].square().mean()
            + output["location_logits"].square().mean()
            + output["mechanism_logits"].square().mean()
            + output["ood_energy"].square().mean()
            + model.observer_nll(self.x, output, self.valid)
            + model.delay_regularization()
        )
        loss.backward()
        for name in (
            "temporal_branch.input_projection.weight",
            "projector.value_projection.weight",
            "fusion_gate_head.weight",
            "conditional_fault_head.weight",
            "fault_logit_head.weight",
            "observer.delta_head.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_hierarchical_class_and_anomaly_probabilities_are_exactly_coherent(self) -> None:
        output = self._forward(self._model().eval())
        class_probability = torch.softmax(output["class_logits"], dim=-1)
        anomaly_probability = torch.softmax(output["anomaly_logits"], dim=-1)
        conditional_non_fault = torch.softmax(
            output["conditional_non_fault_logits"], dim=-1
        )
        conditional_fault = torch.softmax(
            output["conditional_fault_logits"], dim=-1
        )
        non_fault_indices = torch.tensor([0, 1, 2])
        fault_indices = torch.tensor(list(range(3, 20)))

        torch.testing.assert_close(
            class_probability.index_select(-1, non_fault_indices).sum(dim=-1),
            anomaly_probability[..., 0],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            class_probability.index_select(-1, fault_indices).sum(dim=-1),
            anomaly_probability[..., 1],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            class_probability.index_select(-1, non_fault_indices),
            anomaly_probability[..., :1].expand_as(conditional_non_fault)
            * conditional_non_fault,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            class_probability.index_select(-1, fault_indices),
            anomaly_probability[..., 1:].expand_as(conditional_fault)
            * conditional_fault,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            class_probability.sum(dim=-1),
            torch.ones_like(class_probability[..., 0]),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            output["ood_energy"],
            output["residual_energy"] + output["transport_energy"],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(bool((output["ood_energy"] >= 0.0).all()))

    def test_observer_zero_delta_is_exact_previous_sample_persistence(self) -> None:
        observer = PersistenceSkipObserver(len(self.graph.features), 8).eval()
        self.assertEqual(float(observer.delta_head.weight.detach().abs().sum()), 0.0)
        self.assertEqual(float(observer.delta_head.bias.detach().abs().sum()), 0.0)
        mean, log_variance, observer_valid = observer(self.x, self.valid)
        torch.testing.assert_close(mean[:, 1:], self.x[:, :-1], rtol=0.0, atol=0.0)
        self.assertFalse(bool(observer_valid[:, 0].any()))
        self.assertTrue(bool(observer_valid[:, 1:].all()))
        self.assertTrue(torch.isfinite(log_variance).all())

        # A gap cannot yield a nominal training target at the next sample.
        valid = self.valid.clone()
        valid[:, 11] = False
        _, _, observer_valid = observer(self.x, valid)
        self.assertFalse(bool(observer_valid[:, 11].any()))
        self.assertFalse(bool(observer_valid[:, 12].any()))

    def test_model_is_causal_under_arbitrary_future_perturbations(self) -> None:
        model = self._model().eval()
        cutoff = 27
        original = self._forward(model)
        changed_x = self.x.clone()
        changed_x[:, cutoff:] += 10_000.0
        changed_controls = self.controls.clone()
        changed_controls[:, cutoff:, CONTROL_FIELDS.index("measured_flow_l_h")] = 11_000.0
        changed_controls[:, cutoff:, CONTROL_FIELDS.index("cip_cycle_active")] = 1.0
        changed_dt = self.dt_s.clone()
        changed_dt[:, cutoff:] = 1.7
        changed = self._forward(
            model,
            x=changed_x,
            controls=changed_controls,
            dt_s=changed_dt,
        )
        for name in (
            "class_logits",
            "anomaly_logits",
            "location_logits",
            "mechanism_logits",
            "embedding",
            "ood_energy",
            "nominal_mean",
            "residual",
            "temporal_embedding",
            "physics_embedding",
            "fusion_gate",
            "transport_error",
            "delay_seconds",
        ):
            torch.testing.assert_close(
                original[name][:, :cutoff],
                changed[name][:, :cutoff],
                rtol=0.0,
                atol=1e-6,
                msg=name,
            )

    def test_fusion_gate_stays_in_unit_interval_and_padding_is_zero(self) -> None:
        model = self._model().eval()
        valid = self.valid.clone()
        valid[:, -5:] = False
        output = self._forward(model, valid=valid)
        selected_gate = output["fusion_gate"][valid]
        self.assertTrue(bool((selected_gate > 0.0).all()))
        self.assertTrue(bool((selected_gate < 1.0).all()))
        self.assertEqual(
            float(output["fusion_gate"][:, -5:].detach().abs().sum()), 0.0
        )
        for name in (
            "class_logits",
            "anomaly_logits",
            "location_logits",
            "mechanism_logits",
            "embedding",
            "ood_energy",
        ):
            self.assertEqual(float(output[name][:, -5:].detach().abs().sum()), 0.0)

    def test_observer_can_be_frozen_independently_and_inputs_fail_closed(self) -> None:
        model = self._model()
        model.set_observer_trainable(False)
        self.assertFalse(any(parameter.requires_grad for parameter in model.observer.parameters()))
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in model.temporal_branch.parameters()
            )
        )
        model.set_observer_trainable(True)
        self.assertTrue(all(parameter.requires_grad for parameter in model.observer.parameters()))

        with self.assertRaisesRegex(ValueError, "boolean"):
            self._forward(model, valid=self.valid.to(torch.float32))
        invalid_controls = self.controls.clone()
        invalid_controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = -1.0
        with self.assertRaisesRegex(ValueError, "measured flow"):
            self._forward(model, controls=invalid_controls)


if __name__ == "__main__":
    unittest.main()
