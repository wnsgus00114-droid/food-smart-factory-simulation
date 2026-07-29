#!/usr/bin/env python3
"""Contract and forward tests for the nine FlowTwin-Guard ablations."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from flowtwin_guard.ablation import (
        ABLATION_CHOICES,
        ABLATION_NAMES,
        ABLATION_VERSION,
        STATUS_RELAY_FEATURES,
        AblationConfig,
        StaticRouteGate,
        build_ablation_model,
        get_ablation_config,
    )
    from flowtwin_guard.graph import CONTROL_FIELDS, build_process_graph
    from flowtwin_guard.model import FlowTwinConfig


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch ML dependency is not installed")
class FlowTwinAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260727)
        self.graph = build_process_graph()
        self.config = self._config(self.graph)
        self.uncertainty = torch.tensor(
            [0.2 if known else 0.0 for known in self.graph.sensor_uncertainty_known],
            dtype=torch.float32,
        )

    @staticmethod
    def _config(graph) -> FlowTwinConfig:
        return FlowTwinConfig(
            feature_count=len(graph.features),
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            class_count=20,
            location_count=10,
            mechanism_count=10,
            hidden_dim=8,
            observer_hidden_dim=8,
            graph_layers=1,
            dropout=0.0,
        )

    def _batch(self, *, batch: int = 2, steps: int = 9):
        x = torch.randn(batch, steps, len(self.graph.features))
        controls = torch.zeros(batch, steps, len(CONTROL_FIELDS))
        controls[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 20_000.0
        controls[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 1.0
        controls[..., CONTROL_FIELDS.index("steam_valve")] = 0.5
        controls[..., CONTROL_FIELDS.index("power_good_signal")] = 1.0
        dt_s = torch.ones(batch, steps)
        profile = {
            "nominal_flow_l_h": 20_000.0,
            "nominal_holding_time_s": 18.0,
            "sensor_to_fdv_delay_s": 1.5,
            "post_fdv_residence_time_s": 5.0,
        }
        volume = torch.tensor(
            self.graph.edge_volumes_for_profile(profile), dtype=torch.float32
        ).repeat(batch, 1)
        valid = torch.ones(batch, steps, dtype=torch.bool)
        return x, controls, dt_s, volume, valid

    def test_all_nine_configs_are_canonical_and_versioned(self) -> None:
        self.assertEqual(len(ABLATION_NAMES), 9)
        self.assertEqual(set(ABLATION_CHOICES), {"none", *ABLATION_NAMES})
        for name in ABLATION_CHOICES:
            with self.subTest(name=name):
                config = get_ablation_config(name)
                payload = json.loads(json.dumps(config.to_dict()))
                restored = AblationConfig.from_dict(payload)
                self.assertEqual(restored, config)
                self.assertEqual(restored.ablation_version, ABLATION_VERSION)
        self.assertEqual(get_ablation_config("full").name, "none")
        self.assertEqual(get_ablation_config("baseline").name, "none")

    def test_config_rejects_unknown_version_and_mislabeled_signature(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown ablation"):
            get_ablation_config("not_an_ablation")
        payload = get_ablation_config("no_delay").to_dict()
        payload["ablation_version"] = "99.0.0"
        with self.assertRaisesRegex(ValueError, "unsupported ablation version"):
            AblationConfig.from_dict(payload)
        payload = get_ablation_config("no_delay").to_dict()
        payload["delay_mode"] = "flow_dependent"
        with self.assertRaisesRegex(ValueError, "requires delay_mode"):
            AblationConfig.from_dict(payload)

    def test_no_delay_is_literal_zero_including_return_edge(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "no_delay"
        ).eval()
        batch = self._batch()
        original_volume = batch[3].clone()
        output = model(*batch)
        self.assertTrue(torch.equal(batch[3], original_volume))
        self.assertTrue(torch.count_nonzero(output["delay_seconds"]) == 0)
        self.assertFalse(bool(model.graph_blocks[0].one_step_edge.any()))
        self.assertEqual(
            model.training_options["loss_weight_overrides"]["delay_regularization"],
            0.0,
        )

    def test_fixed_nominal_delay_removes_batch_profile_and_flow_variation(self) -> None:
        spec = get_ablation_config(
            "fixed_nominal_delay", fixed_nominal_flow_l_h=19_500.0
        )
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, spec
        ).eval()
        x, controls, dt_s, volume, valid = self._batch()
        controls[0, :, CONTROL_FIELDS.index("measured_flow_l_h")] = 8_000.0
        controls[1, :, CONTROL_FIELDS.index("measured_flow_l_h")] = 31_000.0
        volume[0] *= 0.25
        volume[1] *= 3.0
        transformed = model.transform_inputs(x, controls, dt_s, volume, valid)
        transformed_controls, transformed_volume = transformed[1], transformed[3]
        # The route gate must retain the measured flow; only the private V/Q
        # denominator inside the delay block is fixed.
        torch.testing.assert_close(transformed_controls, controls)
        torch.testing.assert_close(transformed_volume[0], transformed_volume[1])
        output = model(x, controls, dt_s, volume, valid)
        torch.testing.assert_close(
            output["delay_seconds"][0], output["delay_seconds"][1]
        )
        self.assertEqual(model.graph_blocks[0].max_correction, 0.0)

        stopped = controls[:1].clone()
        stopped[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 0.0
        running = stopped.clone()
        running[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 19_500.0
        stopped_gate = model.graph_blocks[0].route_gate(stopped)
        running_gate = model.graph_blocks[0].route_gate(running)
        for tag in ("L-001", "L-004"):
            edge = self.graph.edge_tags.index(tag)
            self.assertEqual(float(stopped_gate[..., edge].sum()), 0.0)
            self.assertGreater(float(running_gate[..., edge].sum()), 0.0)

    def test_static_route_is_independent_of_observable_mode_signals(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "static_route"
        )
        route_gate = model.graph_blocks[0].route_gate
        self.assertIsInstance(route_gate, StaticRouteGate)
        first = self._batch(batch=1, steps=2)[1]
        second = first.clone()
        second[..., CONTROL_FIELDS.index("measured_flow_l_h")] = 0.0
        second[..., CONTROL_FIELDS.index("cip_cycle_active")] = 1.0
        second[..., CONTROL_FIELDS.index("fdv_position_feedback")] = 0.0
        second[..., CONTROL_FIELDS.index("steam_valve")] = 1.0
        second[..., CONTROL_FIELDS.index("power_good_signal")] = 0.0
        torch.testing.assert_close(route_gate(first), route_gate(second))
        by_tag = dict(zip(self.graph.edge_tags, route_gate(first)[0, 0].tolist()))
        for tag in ("L-008", "L-011", "L-201", "U-101"):
            self.assertEqual(by_tag[tag], 1.0)
        for tag in ("L-204", "C-201"):
            self.assertEqual(by_tag[tag], 0.0)

    def test_single_relation_delays_all_features_and_observable_edges(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "single_relation"
        ).eval()
        self.assertTrue(bool(model.projector.advective_feature.all()))
        self.assertTrue(bool(model.graph_blocks[0].advective_edge.all()))
        output = model(*self._batch(), include_node_embeddings=True)
        self.assertEqual(output["node_embedding"].shape, (2, 9, 22, 8))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_raw_only_removes_observer_residual_and_declares_training_change(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "raw_only"
        ).eval()
        x, _controls, _dt_s, _volume, valid = self._batch()
        mean, log_variance, residual, observer_valid = model.observer_outputs(x, valid)
        self.assertTrue(torch.count_nonzero(mean) == 0)
        self.assertTrue(torch.count_nonzero(log_variance) == 0)
        self.assertTrue(torch.count_nonzero(residual) == 0)
        self.assertFalse(bool(observer_valid[:, 0].any()))
        self.assertFalse(model.training_options["train_nominal_observer"])
        with self.assertRaisesRegex(ValueError, "train_nominal_observer"):
            model.ablation_config.assert_training_protocol(
                train_nominal_observer=True, counterfactual_loss_weight=0.10
            )

    def test_no_uncertainty_uses_an_unnormalized_twin_residual(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "no_uncertainty"
        ).eval()
        self.assertTrue(torch.count_nonzero(model.sensor_uncertainty_normalized) == 0)
        x, _controls, _dt_s, _volume, valid = self._batch()
        mean, _log_variance, residual, observer_valid = model.observer_outputs(x, valid)
        expected = (x - mean) * observer_valid.unsqueeze(-1).to(x.dtype)
        torch.testing.assert_close(residual, expected)
        output = {
            "nominal_mean": mean,
            "nominal_log_variance": _log_variance,
            "observer_valid": observer_valid,
        }
        self.assertAlmostEqual(
            float(model.observer_nll(x, output, valid).detach()),
            float((x - mean)[observer_valid].square().mean().detach()),
            places=6,
        )
        with self.assertRaisesRegex(ValueError, "no-op"):
            build_ablation_model(
                self.graph,
                self.config,
                torch.zeros_like(self.uncertainty),
                "no_uncertainty",
            )

    def test_training_and_evaluation_only_ablation_metadata_fail_closed(self) -> None:
        counterfactual = get_ablation_config("no_counterfactual_loss")
        self.assertEqual(
            counterfactual.training_options["loss_weight_overrides"][
                "counterfactual"
            ],
            0.0,
        )
        self.assertFalse(
            counterfactual.training_options["requires_counterfactual_reference"]
        )
        with self.assertRaisesRegex(ValueError, "counterfactual loss weight"):
            counterfactual.assert_training_protocol(
                train_nominal_observer=True, counterfactual_loss_weight=0.10
            )

        conformal = get_ablation_config("no_conformal")
        self.assertEqual(conformal.evaluation_options["decision_policy"], "argmax_only")
        self.assertFalse(conformal.evaluation_options["abstention_enabled"])
        with self.assertRaisesRegex(ValueError, "conformal_enabled"):
            conformal.assert_evaluation_protocol(conformal_enabled=True)

    def test_no_status_relays_neutralizes_only_the_two_direct_status_inputs(self) -> None:
        model = build_ablation_model(
            self.graph, self.config, self.uncertainty, "no_status_relays"
        )
        x, controls, dt_s, volume, valid = self._batch(batch=1, steps=3)
        x.fill_(1.0)
        controls[..., CONTROL_FIELDS.index("power_good_signal")] = 0.0
        original_x = x.clone()
        original_controls = controls.clone()
        transformed = model.transform_inputs(x, controls, dt_s, volume, valid)
        transformed_x, transformed_controls = transformed[0], transformed[1]
        relay_indices = [self.graph.features.index(name) for name in STATUS_RELAY_FEATURES]
        self.assertTrue(torch.count_nonzero(transformed_x[..., relay_indices]) == 0)
        retained = [
            index for index in range(len(self.graph.features)) if index not in relay_indices
        ]
        torch.testing.assert_close(transformed_x[..., retained], x[..., retained])
        self.assertTrue(
            torch.all(
                transformed_controls[
                    ..., CONTROL_FIELDS.index("power_good_signal")
                ]
                == 1.0
            )
        )
        torch.testing.assert_close(x, original_x)
        torch.testing.assert_close(controls, original_controls)

        # Observer pretraining calls observer_outputs directly, so relay
        # exclusion must also be applied inside that API.
        changed = x.clone()
        changed[..., relay_indices] = 10_000.0
        original_observer = model.observer_outputs(x, valid)
        changed_observer = model.observer_outputs(changed, valid)
        for original, alternative in zip(
            original_observer, changed_observer, strict=True
        ):
            torch.testing.assert_close(original, alternative)

    def test_no_status_relays_rejects_a_feature_set_where_it_is_a_no_op(self) -> None:
        removed = self.graph.features.index("temperature_sensor_quality_ok")
        retained = [
            index for index in range(len(self.graph.features)) if index != removed
        ]
        graph = replace(
            self.graph,
            features=tuple(self.graph.features[index] for index in retained),
            feature_node_indices=tuple(
                self.graph.feature_node_indices[index] for index in retained
            ),
            feature_relations=tuple(
                self.graph.feature_relations[index] for index in retained
            ),
            sensor_standard_uncertainty_raw=tuple(
                self.graph.sensor_standard_uncertainty_raw[index] for index in retained
            ),
            sensor_uncertainty_known=tuple(
                self.graph.sensor_uncertainty_known[index] for index in retained
            ),
        )
        with self.assertRaisesRegex(ValueError, "absent from the graph"):
            build_ablation_model(
                graph,
                self._config(graph),
                torch.zeros(len(graph.features)),
                "no_status_relays",
            )


if __name__ == "__main__":
    unittest.main()
