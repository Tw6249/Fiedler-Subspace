#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from violation_free_soc_core import (
    SQRT2,
    MoxChannelParameters,
    active_topology,
    aggregate_parameter_control,
    centralized_soc_control,
    complete_topology,
    cycle_topology,
    diminishing_resource_step,
    distributed_procrustes_align,
    distributed_ritz_estimates,
    exact_low_basis,
    graph_laplacian,
    initial_hexagon,
    initial_resource_state,
    j_inverse,
    j_map,
    local_conic_model,
    lorentz_margin,
    mox_inspired_rate_and_derivative,
    metropolis_matrix,
    objective,
    performance_edges_and_gradients,
    project_lorentz,
    row_ball_projection,
    state_induced_consensus_matrix,
    terminated_violation_free_control,
    true_barrier_matrix,
    violation_free_control,
)


class ConeMappingTests(unittest.TestCase):
    def test_j_isometry_roundtrip_and_cone_equivalence(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(500):
            raw = rng.normal(size=(2, 2))
            matrix = 0.5 * (raw + raw.T)
            vector = j_map(matrix)
            np.testing.assert_allclose(j_inverse(vector), matrix, atol=1e-13)
            self.assertAlmostEqual(np.linalg.norm(matrix, "fro"), np.linalg.norm(vector), places=13)
            self.assertEqual(
                np.linalg.eigvalsh(matrix)[0] >= -1e-12,
                lorentz_margin(vector) >= -SQRT2 * 1e-12,
            )

    def test_lorentz_projection_is_feasible(self) -> None:
        rng = np.random.default_rng(12)
        for _ in range(500):
            projected = project_lorentz(rng.normal(size=3))
            self.assertGreaterEqual(lorentz_margin(projected), -1e-13)

    def test_row_ball_projection(self) -> None:
        values = np.array([[3.0, 4.0, 0.0], [0.2, 0.1, 0.0]])
        projected, active = row_ball_projection(values, radius=1.0)
        self.assertEqual(active, 1)
        self.assertLessEqual(float(np.max(np.linalg.norm(projected, axis=1))), 1.0 + 1e-13)

    def test_diminishing_step_has_required_exponent(self) -> None:
        first = diminishing_resource_step(0, initial_step=0.004, exponent=0.6)
        later = diminishing_resource_step(99, initial_step=0.004, exponent=0.6)
        self.assertAlmostEqual(first, 0.004)
        self.assertLess(later, first)
        with self.assertRaises(ValueError):
            diminishing_resource_step(0, exponent=0.5)


class SmoothChannelTests(unittest.TestCase):
    def test_mox_inspired_cutoff_is_value_and_slope_continuous(self) -> None:
        parameters = MoxChannelParameters()
        step = 1e-5
        for boundary in [
            parameters.transition_distance,
            parameters.cutoff_distance,
        ]:
            value_left, derivative_left = mox_inspired_rate_and_derivative(
                boundary - step, parameters
            )
            value_right, derivative_right = mox_inspired_rate_and_derivative(
                boundary + step, parameters
            )
            self.assertLess(abs(value_left - value_right), 2e-5)
            self.assertLess(abs(derivative_left - derivative_right), 2e-5)
        value, derivative = mox_inspired_rate_and_derivative(
            parameters.cutoff_distance, parameters
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(derivative, 0.0)

    def test_mox_inspired_rate_decreases_and_has_finite_range(self) -> None:
        parameters = MoxChannelParameters()
        values = [
            mox_inspired_rate_and_derivative(distance, parameters)[0]
            for distance in [5.0, 10.0, 20.0, 25.0, 30.0, 35.0]
        ]
        self.assertTrue(all(first >= second for first, second in zip(values, values[1:])))
        self.assertEqual(values[-2:], [0.0, 0.0])

    def test_regular_hexagon_induces_a_cycle_without_edge_masking(self) -> None:
        n = 6
        radius = 17.5
        angles = 2.0 * np.pi * np.arange(n) / n
        positions = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        pairs = complete_topology(n)
        edges, gradients = performance_edges_and_gradients(
            positions,
            pairs,
            sigma=1.0,
            channel_parameters=MoxChannelParameters(),
        )
        support = {tuple(sorted(edge)) for edge in active_topology(edges)}
        expected = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
        self.assertEqual(support, expected)
        for i, j, weight in edges:
            if (i, j) not in support:
                self.assertEqual(weight, 0.0)
                np.testing.assert_allclose(gradients[(i, j)], 0.0, atol=0.0)

    def test_state_induced_mixing_is_doubly_stochastic_on_same_support(self) -> None:
        n = 6
        radius = 17.5
        angles = 2.0 * np.pi * np.arange(n) / n
        positions = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        edges, _ = performance_edges_and_gradients(
            positions,
            complete_topology(n),
            sigma=1.0,
            channel_parameters=MoxChannelParameters(),
        )
        laplacian = graph_laplacian(n, edges)
        mixing, epsilon = state_induced_consensus_matrix(
            laplacian,
            weighted_degree_bound=n - 1,
        )
        self.assertAlmostEqual(epsilon, 1.0 / (2.0 * (n - 1)))
        np.testing.assert_allclose(mixing, mixing.T, atol=1e-14)
        np.testing.assert_allclose(mixing.sum(axis=0), 1.0, atol=1e-14)
        np.testing.assert_allclose(mixing.sum(axis=1), 1.0, atol=1e-14)
        self.assertGreaterEqual(float(mixing.min()), 0.0)
        active = {tuple(sorted(edge)) for edge in active_topology(edges)}
        mixing_support = {
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if mixing[i, j] > 0.0
        }
        self.assertEqual(mixing_support, active)


class AdditiveBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 6
        self.topology = cycle_topology(self.n)
        self.positions = initial_hexagon()
        self.positions += np.array(
            [[0.08, -0.03], [-0.02, 0.06], [0.04, 0.02], [-0.05, -0.04], [0.01, -0.05], [-0.03, 0.03]]
        )
        self.basis, _, _ = exact_low_basis(self.positions, self.topology, 2.0)
        self.model = local_conic_model(self.basis, self.positions, self.topology, 2.0, 3.0, 0.48)

    def test_local_sum_matches_true_compressed_residual(self) -> None:
        controls = np.random.default_rng(13).normal(size=(self.n, 2))
        local_sum = j_inverse(
            np.sum(np.einsum("nab,nb->na", self.model.A, controls) + self.model.c, axis=0)
        )
        truth, _, _ = true_barrier_matrix(self.positions, controls, self.topology, 2.0, 3.0, 0.48)
        np.testing.assert_allclose(local_sum, truth, atol=1e-12)

    def test_violation_free_resource_transient_and_progress(self) -> None:
        mixing = metropolis_matrix(self.n, self.topology)
        communication_laplacian = np.eye(self.n) - mixing
        nominal = 1.1 * (3.0 * initial_hexagon() - self.positions)
        central, _, _ = centralized_soc_control(self.model, nominal)
        state = initial_resource_state(self.n)
        initial_error = None
        minimum_local = np.inf
        minimum_global = np.inf
        for iteration in range(350):
            controls, state, info = violation_free_control(
                self.model,
                nominal,
                communication_laplacian,
                state,
                resource_step=diminishing_resource_step(iteration),
                resource_gain=1.0,
                resource_radius=10.0,
            )
            error = np.linalg.norm(controls - central)
            if initial_error is None:
                initial_error = error
            minimum_local = min(minimum_local, info["minimum_local_margin"] / SQRT2)
            minimum_global = min(minimum_global, info["estimated_tightened_global_margin"] / SQRT2)
        self.assertGreaterEqual(minimum_local, -1e-9)
        self.assertGreaterEqual(minimum_global, -1e-9)
        self.assertLess(error, 0.05 * initial_error)
        self.assertLess(objective(controls, nominal), objective(nominal + 10.0, nominal))
        self.assertLessEqual(info["maximum_resource_state_norm"], 10.0 + 1e-12)
        self.assertGreaterEqual(info["resource_zero_sum_error"], 0.0)

    def test_terminated_inner_loop_applies_only_the_final_candidate(self) -> None:
        positions = np.array(
            [
                [-15.0, 0.0],
                [-9.0, 1.5],
                [-3.0, -1.0],
                [3.0, 1.0],
                [9.0, -1.5],
                [15.0, 0.0],
            ]
        )
        goals = positions.copy()
        goals[0] = [-32.0, 0.0]
        goals[-1] = [32.0, 0.0]
        topology = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        channel = MoxChannelParameters()
        basis, _, _ = exact_low_basis(
            positions, topology, 1.0, channel
        )
        model = local_conic_model(
            basis,
            positions,
            topology,
            1.0,
            2.0,
            0.4,
            beta_total=0.005,
            channel_parameters=channel,
        )
        mixing = metropolis_matrix(6, topology)
        communication_laplacian = np.eye(6) - mixing
        nominal = 0.15 * (goals - positions)

        first_control, _, first_info = violation_free_control(
            model,
            nominal,
            communication_laplacian,
            initial_resource_state(6),
            resource_step=1e-4,
            resource_gain=1.0,
        )
        final_control, final_state, final_info = (
            terminated_violation_free_control(
                model,
                nominal,
                communication_laplacian,
                initial_resource_state(6),
                residual_tolerance=0.1,
                maximum_inner_iterations=10,
                resource_initial_step=1e-4,
                resource_step_exponent=0.6,
            )
        )

        self.assertGreater(np.linalg.norm(first_control - nominal), 1.0)
        self.assertGreater(first_info["allocation_stationarity_residual"], 0.1)
        np.testing.assert_allclose(final_control, nominal, atol=1e-10)
        self.assertEqual(final_info["inner_iterations"], 2.0)
        self.assertEqual(final_info["inner_converged"], 1.0)
        self.assertLessEqual(final_info["termination_residual"], 0.1)
        resources = communication_laplacian @ final_state.y
        local_values = (
            np.einsum("nab,nb->na", model.A, final_control)
            + model.c
            - resources
        )
        self.assertGreaterEqual(
            min(lorentz_margin(value) for value in local_values), -1e-9
        )

    def test_finite_consensus_baseline_can_violate(self) -> None:
        mixing = metropolis_matrix(self.n, self.topology)
        nominal = 1.1 * (3.0 * initial_hexagon() - self.positions)
        residuals = []
        for rounds in range(1, 10):
            _, _, info = aggregate_parameter_control(self.model, nominal, mixing, rounds)
            residuals.append(info["global_margin"] / SQRT2)
        self.assertLess(min(residuals), -1e-6)

    def test_distributed_ritz_estimate_recovers_lambda2(self) -> None:
        mixing = metropolis_matrix(self.n, self.topology)
        _, lambda2_estimates = distributed_ritz_estimates(
            self.model.H,
            mixing,
            consensus_rounds=100,
        )
        _, true_values, _ = exact_low_basis(
            self.positions, self.topology, 2.0
        )
        np.testing.assert_allclose(
            lambda2_estimates,
            true_values[1],
            atol=1e-11,
        )

    def test_distributed_procrustes_recovers_common_gauge(self) -> None:
        mixing = metropolis_matrix(self.n, self.topology)
        angle = 0.73
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        raw = self.basis @ rotation.T
        aligned, factors, disagreement = distributed_procrustes_align(
            raw, self.basis, mixing, consensus_rounds=100
        )
        np.testing.assert_allclose(aligned, self.basis, atol=1e-11)
        np.testing.assert_allclose(
            factors, np.repeat(rotation[None, :, :], self.n, axis=0), atol=1e-11
        )
        self.assertLess(disagreement, 1e-12)


if __name__ == "__main__":
    unittest.main()
