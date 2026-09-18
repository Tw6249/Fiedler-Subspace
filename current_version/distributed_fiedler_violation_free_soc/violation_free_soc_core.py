#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core routines for violation-free distributed Fiedler-SOC simulations.

The controller in this module is self-contained and deliberately does not
import either of the earlier simulation directories.  Global Laplacians and
their eigendecompositions appear only in functions explicitly named
``benchmark_*`` or ``exact_*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

Array = np.ndarray
TopologyEdge = tuple[int, int]


SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# 2 x 2 PSD <-> three-dimensional Lorentz-cone utilities
# ---------------------------------------------------------------------------


def j_map(matrix: Array) -> Array:
    """Isometric map J: S^2 -> R^3 used in the paper draft."""
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    return np.array(
        [
            (matrix[0, 0] + matrix[1, 1]) / SQRT2,
            (matrix[0, 0] - matrix[1, 1]) / SQRT2,
            SQRT2 * matrix[0, 1],
        ]
    )


def j_inverse(vector: Array) -> Array:
    """Inverse of :func:`j_map`."""
    vector = np.asarray(vector, dtype=float)
    return np.array(
        [
            [(vector[0] + vector[1]) / SQRT2, vector[2] / SQRT2],
            [vector[2] / SQRT2, (vector[0] - vector[1]) / SQRT2],
        ]
    )


def lorentz_margin(vector: Array) -> float:
    """Return t-||x|| for a vector (t,x) in R x R^2."""
    vector = np.asarray(vector, dtype=float)
    return float(vector[0] - np.linalg.norm(vector[1:]))


def project_lorentz(vector: Array) -> Array:
    """Euclidean projection onto Q3 = {(t,x): t >= ||x||}."""
    vector = np.asarray(vector, dtype=float)
    t = float(vector[0])
    x = vector[1:]
    radius = float(np.linalg.norm(x))
    if radius <= t:
        return vector.copy()
    if radius <= -t:
        return np.zeros(3)
    projected_t = 0.5 * (radius + t)
    return np.concatenate(([projected_t], projected_t * x / radius))


def matrix_min_eigenvalue_from_cone(vector: Array) -> float:
    """Minimum eigenvalue of J^{-1}(vector), without an eigensolve."""
    return lorentz_margin(vector) / SQRT2


# ---------------------------------------------------------------------------
# Graph, weights, exact benchmark spectrum, and distributed block iteration
# ---------------------------------------------------------------------------


def cycle_topology(n: int) -> list[TopologyEdge]:
    return [(i, (i + 1) % n) for i in range(n)]


def complete_topology(n: int) -> list[TopologyEdge]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def active_topology(
    edges: Sequence[tuple[int, int, float]],
    *,
    tolerance: float = 0.0,
) -> list[TopologyEdge]:
    """Return the positive-weight support of a weighted edge collection."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    return [(i, j) for i, j, weight in edges if weight > tolerance]


def metropolis_matrix(n: int, topology: Sequence[TopologyEdge]) -> Array:
    adjacency = [set() for _ in range(n)]
    for i, j in topology:
        adjacency[i].add(j)
        adjacency[j].add(i)
    mixing = np.zeros((n, n))
    for i in range(n):
        for j in adjacency[i]:
            mixing[i, j] = 1.0 / (1.0 + max(len(adjacency[i]), len(adjacency[j])))
        mixing[i, i] = 1.0 - np.sum(mixing[i])
    return mixing


def graph_laplacian(n: int, edges: Iterable[tuple[int, int, float]]) -> Array:
    laplacian = np.zeros((n, n))
    for i, j, weight in edges:
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
    return laplacian


def state_induced_consensus_matrix(
    laplacian: Array,
    *,
    consensus_step: float | None = None,
    weighted_degree_bound: float | None = None,
) -> tuple[Array, float]:
    """Construct ``W = I - epsilon_c L`` on the current weighted graph.

    If no step is supplied, the routine uses ``1/(2 d_bar)``.  The bound may
    be a uniform offline degree bound or the current maximum weighted degree.
    This choice makes ``W`` nonnegative, symmetric, and doubly stochastic and
    ensures ``epsilon_c * lambda_max(L) <= 1``.
    """
    laplacian = np.asarray(laplacian, dtype=float)
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian must be square")
    laplacian = 0.5 * (laplacian + laplacian.T)
    current_maximum_degree = float(np.max(np.diag(laplacian)))
    if weighted_degree_bound is None:
        weighted_degree_bound = current_maximum_degree
    weighted_degree_bound = float(weighted_degree_bound)
    if weighted_degree_bound <= 0.0:
        raise ValueError("weighted_degree_bound must be positive")
    if current_maximum_degree > weighted_degree_bound + 1e-10:
        raise ValueError("weighted_degree_bound is smaller than the current degree")

    if consensus_step is None:
        consensus_step = 1.0 / (2.0 * weighted_degree_bound)
    consensus_step = float(consensus_step)
    if not (0.0 < consensus_step <= 1.0 / (2.0 * weighted_degree_bound) + 1e-15):
        raise ValueError("consensus_step must lie in (0, 1/(2 d_bar)]")

    mixing = np.eye(laplacian.shape[0]) - consensus_step * laplacian
    if float(np.min(mixing)) < -1e-10:
        raise ValueError("the requested consensus step produces negative weights")
    mixing[np.abs(mixing) < 1e-15] = 0.0
    return mixing, consensus_step


def weighted_edges_and_gradients(
    positions: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
) -> tuple[list[tuple[int, int, float]], dict[tuple[int, int], Array]]:
    """Smooth distance weights and endpoint-local gradients."""
    edges: list[tuple[int, int, float]] = []
    gradients: dict[tuple[int, int], Array] = {}
    for i, j in topology:
        difference = positions[i] - positions[j]
        weight = float(np.exp(-float(difference @ difference) / sigma**2))
        gradient_i = -(2.0 / sigma**2) * weight * difference
        edges.append((i, j, weight))
        gradients[(i, j)] = gradient_i
        gradients[(j, i)] = -gradient_i
    return edges, gradients


@dataclass(frozen=True)
class MoxChannelParameters:
    """Parameters for a finite-range, twice-smooth wireless-rate surrogate.

    The non-vanishing core is the normalized rate used by Mox, Kumar, and
    Ribeiro, with received power ``P_T K d^{-n}``.  Their piecewise-linear
    cutoff is replaced here by a quintic smoothstep over ``[d_t, d_c]``.
    Absolute powers use mW; distances use m.
    """

    transmit_power_mw: float = 1.0
    hardware_gain: float = 5.01e-6
    noise_power_mw: float = 1.0e-7
    path_loss_exponent: float = 2.52
    transition_distance: float = 18.0
    cutoff_distance: float = 30.0

    def __post_init__(self) -> None:
        if self.transmit_power_mw <= 0.0:
            raise ValueError("transmit_power_mw must be positive")
        if self.hardware_gain <= 0.0 or self.noise_power_mw <= 0.0:
            raise ValueError("channel gain and noise power must be positive")
        if self.path_loss_exponent <= 0.0:
            raise ValueError("path_loss_exponent must be positive")
        if not (0.0 < self.transition_distance < self.cutoff_distance):
            raise ValueError("require 0 < transition_distance < cutoff_distance")


def mox_inspired_rate_and_derivative(
    distance: float,
    parameters: MoxChannelParameters,
) -> tuple[float, float]:
    """Return the smooth finite-range rate and its radial derivative.

    The base model is ``erf(sqrt(P_T K d^{-n}/P_N0))``.  Multiplication by
    a quintic smoothstep gives matching values and first two derivatives at
    the transition and cutoff distances, so the extended radial weight is
    C2 on nonnegative distances.
    """
    distance = float(distance)
    if distance < 0.0:
        raise ValueError("distance must be nonnegative")
    if distance >= parameters.cutoff_distance:
        return 0.0, 0.0
    if distance <= 1e-12:
        return 1.0, 0.0

    exponent = parameters.path_loss_exponent
    amplitude = sqrt(
        parameters.transmit_power_mw
        * parameters.hardware_gain
        / parameters.noise_power_mw
    )
    z = amplitude * distance ** (-0.5 * exponent)
    base_rate = erf(z)
    base_derivative = (
        -exponent * z * exp(-(z**2)) / (sqrt(pi) * distance)
    )

    if distance <= parameters.transition_distance:
        return float(base_rate), float(base_derivative)

    width = parameters.cutoff_distance - parameters.transition_distance
    s = (distance - parameters.transition_distance) / width
    cutoff = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
    cutoff_derivative = (-30.0 * s**2 + 60.0 * s**3 - 30.0 * s**4) / width
    rate = base_rate * cutoff
    derivative = base_derivative * cutoff + base_rate * cutoff_derivative
    return float(rate), float(derivative)


def mox_inspired_edges_and_gradients(
    positions: Array,
    topology: Sequence[TopologyEdge],
    parameters: MoxChannelParameters,
) -> tuple[list[tuple[int, int, float]], dict[tuple[int, int], Array]]:
    """Mox-inspired finite-range weights and endpoint-local gradients."""
    edges: list[tuple[int, int, float]] = []
    gradients: dict[tuple[int, int], Array] = {}
    for i, j in topology:
        difference = positions[i] - positions[j]
        distance = float(np.linalg.norm(difference))
        weight, radial_derivative = mox_inspired_rate_and_derivative(
            distance, parameters
        )
        if distance <= 1e-12:
            gradient_i = np.zeros_like(difference)
        else:
            gradient_i = radial_derivative * difference / distance
        edges.append((i, j, weight))
        gradients[(i, j)] = gradient_i
        gradients[(j, i)] = -gradient_i
    return edges, gradients


def performance_edges_and_gradients(
    positions: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
    channel_parameters: MoxChannelParameters | None = None,
) -> tuple[list[tuple[int, int, float]], dict[tuple[int, int], Array]]:
    if channel_parameters is None:
        return weighted_edges_and_gradients(positions, topology, sigma)
    return mox_inspired_edges_and_gradients(
        positions, topology, channel_parameters
    )


def exact_low_basis(
    positions: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
    channel_parameters: MoxChannelParameters | None = None,
) -> tuple[Array, Array, Array]:
    """Centralized truth used only by offline/oracle benchmarks."""
    edges, _ = performance_edges_and_gradients(
        positions, topology, sigma, channel_parameters
    )
    laplacian = graph_laplacian(len(positions), edges)
    values, vectors = np.linalg.eigh(laplacian)
    return vectors[:, 1:3], values, laplacian


def average_consensus(samples: Array, mixing: Array, rounds: int) -> Array:
    estimates = np.asarray(samples, dtype=float).copy()
    for _ in range(rounds):
        estimates = np.tensordot(mixing, estimates, axes=(1, 0))
    return estimates


def distributed_ritz_estimates(
    local_ritz_contributions: Array,
    mixing: Array,
    consensus_rounds: int,
) -> tuple[Array, Array]:
    """Estimate the global compressed Ritz matrix and lambda2 at every node.

    Node ``i`` starts from its locally computable contribution ``H_i``.  An
    average-consensus estimate is multiplied by ``n`` so that

        H_hat_i ~= sum_l H_l = X.T @ L @ X,
        lambda2_hat_i = lambda_min(H_hat_i).

    This scalar estimate is a point estimate for monitoring; robust safety is
    still enforced by the tightened matrix/SOC constraint rather than by
    treating ``lambda2_hat_i`` as a certified lower bound.
    """
    local_ritz_contributions = np.asarray(local_ritz_contributions, dtype=float)
    n = local_ritz_contributions.shape[0]
    estimates = n * average_consensus(
        local_ritz_contributions, mixing, consensus_rounds
    )
    estimates = 0.5 * (estimates + np.swapaxes(estimates, -1, -2))
    lambda2_estimates = np.linalg.eigvalsh(estimates)[:, 0]
    return estimates, lambda2_estimates


def inverse_sqrt_psd(matrix: Array, floor: float = 1e-10) -> Array:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def distributed_block_step(
    node_basis: Array,
    edges: Sequence[tuple[int, int, float]],
    mixing: Array,
    diffusion_step: float,
    consensus_rounds: int,
) -> Array:
    """One row-distributed block orthogonal iteration with finite consensus.

    Unlike the previous simulation, this version follows draft (2)'s
    normalization X.T X = I by converting average-consensus estimates into
    estimates of the global sums.
    """
    n = len(node_basis)
    diffused = node_basis.copy()
    for i, j, weight in edges:
        difference = node_basis[i] - node_basis[j]
        diffused[i] -= diffusion_step * weight * difference
        diffused[j] += diffusion_step * weight * difference

    mean_estimates = average_consensus(diffused, mixing, consensus_rounds)
    centered = diffused - mean_estimates
    gram_samples = np.einsum("ni,nj->nij", centered, centered)
    gram_sum_estimates = n * average_consensus(gram_samples, mixing, consensus_rounds)

    updated = np.empty_like(node_basis)
    for i in range(n):
        updated[i] = centered[i] @ inverse_sqrt_psd(gram_sum_estimates[i])
    return updated


def projector_error(estimated_basis: Array, true_basis: Array) -> float:
    """Grassmann sine error after globally orthonormalizing the estimate."""
    centered = estimated_basis - np.mean(estimated_basis, axis=0, keepdims=True)
    q_est, _ = np.linalg.qr(centered)
    singular_values = np.linalg.svd(q_est[:, :2].T @ true_basis, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - np.min(singular_values) ** 2)))


def align_basis(estimated_basis: Array, reference_basis: Array) -> tuple[Array, Array]:
    """Return estimated_basis Q and the best orthogonal two-dimensional Q."""
    left, _, right_t = np.linalg.svd(estimated_basis.T @ reference_basis)
    q = left @ right_t
    return estimated_basis @ q, q


def distributed_procrustes_align(
    raw_basis: Array,
    reference_basis: Array,
    mixing: Array,
    consensus_rounds: int,
) -> tuple[Array, Array, float]:
    """Align a row-distributed basis to its previous gauge.

    Node ``i`` starts from the locally available cross sample
    ``raw_basis[i]^T reference_basis[i]``.  Finite-round average consensus
    estimates the global two-by-two cross matrix at every node.  Each node
    then applies the corresponding orthogonal Procrustes factor to its row.
    The returned disagreement is the largest Frobenius distance between a
    local factor and their Euclidean mean; it is a diagnostic, not a
    centralized input to the controller.
    """
    raw_basis = np.asarray(raw_basis, dtype=float)
    reference_basis = np.asarray(reference_basis, dtype=float)
    if raw_basis.shape != reference_basis.shape or raw_basis.shape[1] != 2:
        raise ValueError("Procrustes alignment requires two equally shaped N-by-2 bases.")

    n = len(raw_basis)
    cross_samples = np.einsum("ni,nj->nij", raw_basis, reference_basis)
    cross_estimates = n * average_consensus(cross_samples, mixing, consensus_rounds)
    factors = np.empty((n, 2, 2))
    aligned = np.empty_like(raw_basis)
    for i in range(n):
        left, _, right_t = np.linalg.svd(cross_estimates[i])
        factors[i] = left @ right_t
        aligned[i] = raw_basis[i] @ factors[i]

    factor_mean = np.mean(factors, axis=0)
    disagreement = float(
        np.max(np.linalg.norm(factors - factor_mean, axis=(1, 2)))
    )
    return aligned, factors, disagreement


def row_ball_projection(values: Array, radius: float) -> tuple[Array, int]:
    """Project each row independently onto a Euclidean ball."""
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("The auxiliary-potential projection radius must be positive and finite.")
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    scales = np.ones_like(norms)
    active = norms > radius
    scales[active] = radius / norms[active]
    return values * scales[:, None], int(np.sum(active))


def diminishing_resource_step(
    iteration: int,
    initial_step: float = 0.004,
    exponent: float = 0.6,
) -> float:
    """Square-summable but nonsummable auxiliary-allocation step schedule."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if not (0.5 < exponent <= 1.0):
        raise ValueError("The exponent must lie in (1/2, 1].")
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    return float(initial_step / (iteration + 1.0) ** exponent)


# ---------------------------------------------------------------------------
# Local additive spectral barrier model
# ---------------------------------------------------------------------------


@dataclass
class ConicModel:
    A: Array  # (n, 3, 2), local control-to-cone maps
    c: Array  # (n, 3), local constant terms (including tightening)
    c_untightened: Array  # (n, 3)
    H: Array  # (n, 2, 2), local Ritz contributions
    B: Array  # (n, 2, 2, 2), B[i, coordinate, :, :]
    beta: Array  # (n,), local residual tightening


def local_conic_model(
    node_basis: Array,
    positions: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
    alpha: float,
    lambda_des: float,
    beta_total: float = 0.0,
    channel_parameters: MoxChannelParameters | None = None,
) -> ConicModel:
    """Construct Psi_i(u_i) and its exact Q3 representation locally."""
    n = len(positions)
    edges, gradients = performance_edges_and_gradients(
        positions, topology, sigma, channel_parameters
    )
    H = np.zeros((n, 2, 2))
    B = np.zeros((n, 2, 2, 2))

    for i, j, weight in edges:
        difference = node_basis[i] - node_basis[j]
        edge_matrix = np.outer(difference, difference)
        H[i] += 0.5 * weight * edge_matrix
        H[j] += 0.5 * weight * edge_matrix
        for coordinate in range(2):
            B[i, coordinate] += gradients[(i, j)][coordinate] * edge_matrix
            B[j, coordinate] += gradients[(j, i)][coordinate] * edge_matrix

    beta = np.full(n, beta_total / n)
    A = np.zeros((n, 3, 2))
    c = np.zeros((n, 3))
    c_untightened = np.zeros((n, 3))
    for i in range(n):
        for coordinate in range(2):
            A[i, :, coordinate] = j_map(B[i, coordinate])
        constant_matrix = alpha * (H[i] - (lambda_des / n) * np.eye(2))
        c_untightened[i] = j_map(constant_matrix)
        c[i] = j_map(constant_matrix - beta[i] * np.eye(2))

    return ConicModel(A=A, c=c, c_untightened=c_untightened, H=H, B=B, beta=beta)


def spectral_model_error_bound(
    estimated_basis: Array,
    true_basis: Array,
    positions: Array,
    controls: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
    alpha: float,
    channel_parameters: MoxChannelParameters | None = None,
) -> tuple[float, Array, float]:
    """A posteriori bound induced by a global Procrustes row error.

    This routine is deliberately diagnostic: ``true_basis`` is centralized
    offline truth and must not be used by the distributed controller.  It
    evaluates a sharper offline diagnostic than the manuscript's sufficient
    transfer certificate because it retains the exact ``1/2`` multiplier:

        Delta_xi(epsilon_X) (alpha d_i^w / 2 + ||g_i|| ||u_i||)

    for every node, where ``epsilon_X`` is the aligned spectral-norm row
    error.  A certified online tightening requires an a priori estimator
    bound in place of this offline value.  Even this sharper quantity is
    diagnostic rather than an online certificate.
    """
    aligned, _ = align_basis(estimated_basis, true_basis)
    epsilon_x = float(np.linalg.norm(aligned - true_basis, ord=2))
    delta_xi = 8.0 * epsilon_x + 4.0 * epsilon_x**2

    n = len(positions)
    weighted_degree_half = np.zeros(n)
    gradient_components = np.zeros((n, 2))
    edges, gradients = performance_edges_and_gradients(
        positions, topology, sigma, channel_parameters
    )
    for i, j, weight in edges:
        weighted_degree_half[i] += 0.5 * weight
        weighted_degree_half[j] += 0.5 * weight
        gradient_components[i] += np.abs(gradients[(i, j)])
        gradient_components[j] += np.abs(gradients[(j, i)])

    local_bounds = delta_xi * (
        alpha * weighted_degree_half
        + np.linalg.norm(gradient_components, axis=1)
        * np.linalg.norm(controls, axis=1)
    )
    return epsilon_x, local_bounds, float(np.sum(local_bounds))


def true_barrier_matrix(
    positions: Array,
    controls: Array,
    topology: Sequence[TopologyEdge],
    sigma: float,
    alpha: float,
    lambda_des: float,
    channel_parameters: MoxChannelParameters | None = None,
) -> tuple[Array, Array, Array]:
    """Offline true compressed derivative residual and spectrum."""
    basis, values, laplacian = exact_low_basis(
        positions, topology, sigma, channel_parameters
    )
    edges, gradients = performance_edges_and_gradients(
        positions, topology, sigma, channel_parameters
    )
    laplacian_rate = np.zeros_like(laplacian)
    for i, j, _ in edges:
        rate = float(gradients[(i, j)] @ (controls[i] - controls[j]))
        laplacian_rate[i, i] += rate
        laplacian_rate[j, j] += rate
        laplacian_rate[i, j] -= rate
        laplacian_rate[j, i] -= rate
    residual = (
        basis.T @ laplacian_rate @ basis
        + alpha * (basis.T @ laplacian @ basis - lambda_des * np.eye(2))
    )
    return 0.5 * (residual + residual.T), values, basis


# ---------------------------------------------------------------------------
# Three-variable dual solution and conic controllers
# ---------------------------------------------------------------------------


@dataclass
class SocSolution:
    control: Array
    dual: Array
    cone_value: Array
    margin: float
    iterations: int
    gradient_mapping: float
    complementarity: float


def solve_cone_dual(
    K: Array,
    m: Array,
    initial_dual: Array | None = None,
    tolerance: float = 2e-11,
    max_iterations: int = 6000,
) -> tuple[Array, int, float]:
    """Minimize 0.5 z'Kz + m'z over z in Q3.

    A nonzero optimum of this three-dimensional problem normally lies on the
    Lorentz-cone boundary and can be parameterized as
    ``z=tau*[1, cos(theta), sin(theta)]``.  A coarse periodic search followed
    by a bounded scalar refinement is both faster and more accurate than a
    long generic projected-gradient loop.  The interior stationary point is
    checked separately.  ``max_iterations`` remains in the signature for API
    compatibility and as a guard for the rare projected-gradient fallback.
    """
    K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    m = np.asarray(m, dtype=float)
    if lorentz_margin(m) >= -tolerance:
        return np.zeros(3), 1, 0.0

    largest = max(float(np.linalg.eigvalsh(K)[-1]), 1e-12)
    step = 0.98 / largest

    candidates: list[Array] = []
    pseudo = -np.linalg.pinv(K, rcond=1e-13) @ m
    if (
        np.linalg.norm(K @ pseudo + m) <= 1e-9 * (1.0 + np.linalg.norm(m))
        and lorentz_margin(pseudo) >= -1e-10
    ):
        candidates.append(project_lorentz(pseudo))

    sample_count = 180
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    directions = np.column_stack((np.ones(sample_count), np.cos(angles), np.sin(angles)))
    linear = directions @ m
    quadratic = np.einsum("ni,ij,nj->n", directions, K, directions)
    values = np.full(sample_count, np.inf)
    regular = quadratic > 1e-14
    active = regular & (linear < 0.0)
    values[active] = -0.5 * linear[active] ** 2 / quadratic[active]
    best_index = int(np.argmin(values))

    if np.isfinite(values[best_index]):
        spacing = 2.0 * np.pi / sample_count

        def boundary_objective(theta: float) -> float:
            direction = np.array([1.0, np.cos(theta), np.sin(theta)])
            denominator = float(direction @ K @ direction)
            numerator = float(m @ direction)
            if denominator <= 1e-15:
                return -np.inf if numerator < 0.0 else 0.0
            return -0.5 * min(numerator, 0.0) ** 2 / denominator

        refined = minimize_scalar(
            boundary_objective,
            bounds=(angles[best_index] - spacing, angles[best_index] + spacing),
            method="bounded",
            options={"xatol": 1e-14, "maxiter": 100},
        )
        theta = float(refined.x)
        direction = np.array([1.0, np.cos(theta), np.sin(theta)])
        denominator = float(direction @ K @ direction)
        numerator = float(m @ direction)
        tau = max(0.0, -numerator / max(denominator, 1e-15))
        candidates.append(tau * direction)
        scalar_evaluations = int(getattr(refined, "nfev", 0))
    else:
        scalar_evaluations = 0

    if candidates:
        objectives = [0.5 * float(z @ K @ z) + float(m @ z) for z in candidates]
        z = candidates[int(np.argmin(objectives))]
        gradient = K @ z + m
        mapping = float(np.linalg.norm(project_lorentz(z - step * gradient) - z) / step)
        if mapping <= 1e-10:
            return z, sample_count + scalar_evaluations, mapping

    # Rare numerical fallback (for a nearly degenerate K or a missed angular
    # minimum).  Warm starts are useful here when the auxiliary-allocation flow is active.
    z = np.zeros(3) if initial_dual is None else project_lorentz(initial_dual)
    mapping = np.inf
    for iteration in range(1, max_iterations + 1):
        gradient = K @ z + m
        z_next = project_lorentz(z - step * gradient)
        mapping = float(np.linalg.norm(z_next - z) / step)
        z = z_next
        if mapping <= tolerance:
            break
    return z, iteration, mapping


def solve_local_soc(
    A: Array,
    c: Array,
    resource: Array,
    nominal_control: Array,
    initial_dual: Array | None = None,
) -> SocSolution:
    """Euclidean projection of nominal_control onto one affine SOC set."""
    m = A @ nominal_control + c - resource
    K = A @ A.T
    dual, iterations, mapping = solve_cone_dual(K, m, initial_dual)
    control = nominal_control + A.T @ dual
    cone_value = A @ control + c - resource

    # A nearly singular dual Hessian can make projected-gradient fallback
    # converge very slowly in the dual null direction.  If this leaves a
    # visible primal cone error, finish the two-variable primal projection by
    # SLSQP.  A locally generated Slater direction supplies a feasible start;
    # no global information or large conic solver is involved.
    if lorentz_margin(cone_value) < -1e-9:
        angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        cone_directions = directions @ A.T
        direction_margins = cone_directions[:, 0] - np.linalg.norm(cone_directions[:, 1:], axis=1)
        best = int(np.argmax(direction_margins))
        slater_direction = directions[best]
        slater_cone_direction = A @ slater_direction
        if direction_margins[best] <= 1e-12:
            raise RuntimeError("The local control map has no numerically detectable Slater direction.")

        scale = 1.0
        while lorentz_margin(cone_value + scale * slater_cone_direction) < 1e-10:
            scale *= 2.0
            if scale > 1e12:
                raise RuntimeError("Unable to construct a feasible local SOCP start.")
        feasible_start = control + scale * slater_direction

        def primal_objective(candidate: Array) -> float:
            difference = candidate - nominal_control
            return 0.5 * float(difference @ difference)

        def primal_jacobian(candidate: Array) -> Array:
            return candidate - nominal_control

        def cone_constraint(candidate: Array) -> float:
            return lorentz_margin(A @ candidate + c - resource)

        def cone_constraint_jacobian(candidate: Array) -> Array:
            value = A @ candidate + c - resource
            radius = float(np.linalg.norm(value[1:]))
            if radius <= 1e-14:
                return A[0].copy()
            return A[0] - (value[1:] / radius) @ A[1:]

        refined = minimize(
            primal_objective,
            feasible_start,
            jac=primal_jacobian,
            constraints={"type": "ineq", "fun": cone_constraint, "jac": cone_constraint_jacobian},
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 150, "disp": False},
        )
        if not refined.success and cone_constraint(refined.x) < -1e-8:
            raise RuntimeError(f"Local SOCP refinement failed: {refined.message}")
        control = np.asarray(refined.x, dtype=float)
        cone_value = A @ control + c - resource
        iterations += int(getattr(refined, "nit", 0))

        # Recover the Q3 multiplier from stationarity.  On a smooth cone
        # boundary, z=lambda*[1,-q_tail/||q_tail||].
        radius = float(np.linalg.norm(cone_value[1:]))
        if radius > 1e-10:
            dual_direction = np.concatenate(([1.0], -cone_value[1:] / radius))
            stationarity_direction = A.T @ dual_direction
            denominator = float(stationarity_direction @ stationarity_direction)
            if denominator > 1e-16:
                multiplier = max(
                    0.0,
                    float((control - nominal_control) @ stationarity_direction) / denominator,
                )
                dual = multiplier * dual_direction
                gradient = K @ dual + m
                largest = max(float(np.linalg.eigvalsh(K)[-1]), 1e-12)
                step = 0.98 / largest
                mapping = float(
                    np.linalg.norm(project_lorentz(dual - step * gradient) - dual) / step
                )

    return SocSolution(
        control=control,
        dual=dual,
        cone_value=cone_value,
        margin=lorentz_margin(cone_value),
        iterations=iterations,
        gradient_mapping=mapping,
        complementarity=float(abs(dual @ cone_value)),
    )


def centralized_soc_control(
    model: ConicModel,
    nominal_controls: Array,
    initial_dual: Array | None = None,
) -> tuple[Array, Array, dict[str, float]]:
    """Centralized benchmark for the summed SOC constraint."""
    n = len(nominal_controls)
    A_global = np.concatenate([model.A[i] for i in range(n)], axis=1)
    nominal_global = nominal_controls.reshape(-1)
    c_global = np.sum(model.c, axis=0)
    solution = solve_local_soc(
        A_global,
        c_global,
        np.zeros(3),
        nominal_global,
        initial_dual,
    )
    return solution.control.reshape(n, 2), solution.dual, {
        "global_margin": solution.margin,
        "iterations": float(solution.iterations),
        "complementarity": solution.complementarity,
    }


def aggregate_parameter_control(
    model: ConicModel,
    nominal_controls: Array,
    mixing: Array,
    consensus_rounds: int,
) -> tuple[Array, Array, dict[str, float]]:
    """Previous-style finite-consensus parameter aggregation baseline."""
    n = len(nominal_controls)
    local_K = np.einsum("nac,nbc->nab", model.A, model.A)
    local_m = np.einsum("nab,nb->na", model.A, nominal_controls) + model.c
    K_estimates = n * average_consensus(local_K, mixing, consensus_rounds)
    m_estimates = n * average_consensus(local_m, mixing, consensus_rounds)

    controls = nominal_controls.copy()
    duals = np.zeros((n, 3))
    for i in range(n):
        duals[i], _, _ = solve_cone_dual(K_estimates[i], m_estimates[i])
        controls[i] += model.A[i].T @ duals[i]

    global_value = np.sum(
        np.einsum("nab,nb->na", model.A, controls) + model.c,
        axis=0,
    )
    disagreement = np.max(np.linalg.norm(duals - np.mean(duals, axis=0), axis=1))
    return controls, duals, {
        "global_margin": lorentz_margin(global_value),
        "dual_disagreement": float(disagreement),
    }


@dataclass
class ResourceState:
    y: Array
    duals: Array


def initial_resource_state(n: int) -> ResourceState:
    return ResourceState(y=np.zeros((n, 3)), duals=np.zeros((n, 3)))


def violation_free_control(
    model: ConicModel,
    nominal_controls: Array,
    communication_laplacian: Array,
    state: ResourceState,
    resource_step: float,
    resource_gain: float,
    resource_radius: float | None = None,
) -> tuple[Array, ResourceState, dict[str, float]]:
    """One local-SOCP auxiliary-allocation step of the distributed method."""
    n = len(nominal_controls)
    resources = communication_laplacian @ state.y
    controls = np.empty_like(nominal_controls)
    duals = np.empty((n, 3))
    local_margins = np.empty(n)
    iterations = np.empty(n)
    complementarity = np.empty(n)
    local_solve_times_ms = np.empty(n)

    for i in range(n):
        solve_started = perf_counter()
        solution = solve_local_soc(
            model.A[i],
            model.c[i],
            resources[i],
            nominal_controls[i],
            state.duals[i],
        )
        local_solve_times_ms[i] = 1e3 * (perf_counter() - solve_started)
        controls[i] = solution.control
        duals[i] = solution.dual
        local_margins[i] = solution.margin
        iterations[i] = solution.iterations
        complementarity[i] = solution.complementarity

    estimated_global_value = np.sum(
        np.einsum("nab,nb->na", model.A, controls) + model.c,
        axis=0,
    )
    untightened_global_value = np.sum(
        np.einsum("nab,nb->na", model.A, controls) + model.c_untightened,
        axis=0,
    )

    allocation_residuals = communication_laplacian @ duals
    y_next = state.y - resource_step * resource_gain * allocation_residuals
    projection_count = 0
    if resource_radius is not None:
        y_next, projection_count = row_ball_projection(y_next, resource_radius)
    next_state = ResourceState(y=y_next, duals=duals)
    disagreement = np.max(np.linalg.norm(duals - np.mean(duals, axis=0), axis=1))
    next_resources = communication_laplacian @ y_next

    return controls, next_state, {
        "minimum_local_margin": float(np.min(local_margins)),
        "estimated_tightened_global_margin": lorentz_margin(estimated_global_value),
        "estimated_untightened_global_margin": lorentz_margin(untightened_global_value),
        "dual_disagreement": float(disagreement),
        "allocation_stationarity_residual": float(np.linalg.norm(allocation_residuals)),
        "maximum_local_allocation_residual": float(
            np.max(np.linalg.norm(allocation_residuals, axis=1))
        ),
        "maximum_dual_norm": float(np.max(np.linalg.norm(duals, axis=1))),
        "mean_local_iterations": float(np.mean(iterations)),
        "maximum_complementarity": float(np.max(complementarity)),
        "median_local_solve_time_ms": float(np.median(local_solve_times_ms)),
        "maximum_local_solve_time_ms": float(np.max(local_solve_times_ms)),
        "total_local_solve_time_ms": float(np.sum(local_solve_times_ms)),
        "resource_zero_sum_error": float(np.linalg.norm(np.sum(next_resources, axis=0))),
        "maximum_resource_state_norm": float(np.max(np.linalg.norm(y_next, axis=1))),
        "resource_projection_count": float(projection_count),
    }


def terminated_violation_free_control(
    model: ConicModel,
    nominal_controls: Array,
    communication_laplacian: Array,
    state: ResourceState,
    *,
    residual_tolerance: float,
    maximum_inner_iterations: int,
    resource_initial_step: float,
    resource_step_exponent: float,
    resource_gain: float = 1.0,
    resource_radius: float | None = None,
    minimum_inner_iterations: int = 1,
) -> tuple[Array, ResourceState, dict[str, float]]:
    """Run a frozen-model auxiliary-allocation loop before applying control.

    The plant state, nominal control, and conic model are fixed throughout the
    loop.  Every returned candidate is globally feasible for the summed exact
    frozen-model cone whenever all local SOCPs are feasible.  The stopping
    residual is ``||(L_c kron I_3) z||``.  The returned warm start contains the
    auxiliary potential at which the final control was computed, rather than
    the untested potential after one additional resource update.
    """
    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be nonnegative")
    if maximum_inner_iterations < 1:
        raise ValueError("maximum_inner_iterations must be positive")
    if not (1 <= minimum_inner_iterations <= maximum_inner_iterations):
        raise ValueError(
            "minimum_inner_iterations must lie between one and maximum_inner_iterations"
        )

    current_state = ResourceState(y=state.y.copy(), duals=state.duals.copy())
    maximum_dual_norm = 0.0
    maximum_resource_state_norm = float(
        np.max(np.linalg.norm(current_state.y, axis=1))
    )
    maximum_allocation_norm = float(
        np.max(
            np.linalg.norm(communication_laplacian @ current_state.y, axis=1)
        )
    )
    total_projection_count = 0.0
    total_local_solve_time_ms = 0.0
    maximum_local_solve_time_ms = 0.0
    converged = False

    for inner_index in range(maximum_inner_iterations):
        step = diminishing_resource_step(
            inner_index,
            initial_step=resource_initial_step,
            exponent=resource_step_exponent,
        )
        controls, next_state, step_info = violation_free_control(
            model,
            nominal_controls,
            communication_laplacian,
            current_state,
            resource_step=step,
            resource_gain=resource_gain,
            resource_radius=resource_radius,
        )
        applied_state = ResourceState(
            y=current_state.y.copy(),
            duals=next_state.duals.copy(),
        )
        maximum_dual_norm = max(
            maximum_dual_norm, step_info["maximum_dual_norm"]
        )
        maximum_resource_state_norm = max(
            maximum_resource_state_norm,
            step_info["maximum_resource_state_norm"],
        )
        maximum_allocation_norm = max(
            maximum_allocation_norm,
            float(
                np.max(
                    np.linalg.norm(
                        communication_laplacian @ next_state.y, axis=1
                    )
                )
            ),
        )
        total_projection_count += step_info["resource_projection_count"]
        total_local_solve_time_ms += step_info["total_local_solve_time_ms"]
        maximum_local_solve_time_ms = max(
            maximum_local_solve_time_ms,
            step_info["maximum_local_solve_time_ms"],
        )

        inner_iterations = inner_index + 1
        if (
            inner_iterations >= minimum_inner_iterations
            and step_info["allocation_stationarity_residual"]
            <= residual_tolerance
        ):
            converged = True
            break
        current_state = next_state

    final_info = dict(step_info)
    final_info.update(
        {
            "inner_iterations": float(inner_iterations),
            "inner_converged": float(converged),
            "termination_residual": step_info[
                "allocation_stationarity_residual"
            ],
            "termination_maximum_local_residual": step_info[
                "maximum_local_allocation_residual"
            ],
            "maximum_inner_dual_norm": float(maximum_dual_norm),
            "maximum_inner_resource_state_norm": float(
                maximum_resource_state_norm
            ),
            "maximum_inner_allocation_norm": float(maximum_allocation_norm),
            "total_inner_resource_projection_count": float(
                total_projection_count
            ),
            "total_local_solve_time_ms": float(total_local_solve_time_ms),
            "maximum_inner_local_solve_time_ms": float(
                maximum_local_solve_time_ms
            ),
            "final_resource_step": float(step),
        }
    )
    return controls, applied_state, final_info


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def objective(controls: Array, nominal_controls: Array) -> float:
    return 0.5 * float(np.sum((controls - nominal_controls) ** 2))


def gamma_star(local_B: Array, angular_samples: int = 1440) -> float:
    """Numerical max_{||d||=1} lambda_min(d1 B1+d2 B2)."""
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    matrices = cosine[:, None, None] * local_B[0] + sine[:, None, None] * local_B[1]
    a = matrices[:, 0, 0]
    b = matrices[:, 0, 1]
    c = matrices[:, 1, 1]
    minimum = 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4.0 * b**2))
    return float(np.max(minimum))


def all_gamma_stars(model: ConicModel, angular_samples: int = 1440) -> Array:
    return np.array([gamma_star(model.B[i], angular_samples) for i in range(len(model.B))])


def initial_hexagon(n: int = 6) -> Array:
    angles = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angles), np.sin(angles)))


def scenario_goals(positions: Array, scenario: str) -> Array:
    if scenario == "radial":
        return 3.0 * positions
    if scenario == "anisotropic":
        goals = positions.copy()
        goals[:, 0] *= 3.0
        goals[:, 1] *= 2.4
        return goals
    raise ValueError(f"Unknown scenario: {scenario}")
