#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State-induced single-graph experiments used by the current manuscript.

Every robot pair is evaluated by the same compact-support channel model.
Positive weights define the active communication graph, and that graph is
used for spectral diffusion, finite-round consensus, and auxiliary
allocation.  No persistent cycle edge set is supplied to any experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from violation_free_soc_core import (
    SQRT2,
    MoxChannelParameters,
    ResourceState,
    active_topology,
    all_gamma_stars,
    aggregate_parameter_control,
    average_consensus,
    centralized_soc_control,
    complete_topology,
    diminishing_resource_step,
    distributed_block_step,
    exact_low_basis,
    graph_laplacian,
    initial_resource_state,
    local_conic_model,
    objective,
    performance_edges_and_gradients,
    state_induced_consensus_matrix,
    true_barrier_matrix,
    violation_free_control,
)

Array = np.ndarray


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.45,
    }
)

COLORS = {
    "block": "#D55E00",
    "single": "#0072B2",
    "truth": "#222222",
    "lambda3": "#777777",
    "nominal": "#999999",
    "centralized": "#0072B2",
    "aggregate": "#E69F00",
    "allocation": "#D55E00",
    "safe": "#009E73",
    "secondary": "#56B4E9",
    "yang": "#009E73",
    "zhang": "#CC79A7",
}


def save_both(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.pdf")
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(figure)


def channel_graph(
    positions: Array,
    channel: MoxChannelParameters,
) -> tuple[list[tuple[int, int, float]], Array, Array, float]:
    """Return channel edges, Laplacian, and same-graph consensus matrix."""
    n = len(positions)
    pairs = complete_topology(n)
    edges, _ = performance_edges_and_gradients(
        positions,
        pairs,
        sigma=1.0,
        channel_parameters=channel,
    )
    laplacian = graph_laplacian(n, edges)
    mixing, epsilon_c = state_induced_consensus_matrix(
        laplacian,
        weighted_degree_bound=float(n - 1),
    )
    return edges, laplacian, mixing, epsilon_c


def local_ritz_contributions(node_basis: Array, edges: list[tuple[int, int, float]]) -> Array:
    n, rank = node_basis.shape
    contributions = np.zeros((n, rank, rank))
    for i, j, weight in edges:
        difference = node_basis[i] - node_basis[j]
        edge_matrix = np.outer(difference, difference)
        contributions[i] += 0.5 * weight * edge_matrix
        contributions[j] += 0.5 * weight * edge_matrix
    return contributions


def grassmann_sine_error(estimated: Array, truth: Array) -> float:
    centered = estimated - np.mean(estimated, axis=0, keepdims=True)
    q_est, _ = np.linalg.qr(centered, mode="reduced")
    singular_values = np.linalg.svd(q_est.T @ truth, compute_uv=False)
    squared = max(0.0, truth.shape[1] - float(np.sum(singular_values**2)))
    return float(np.sqrt(squared))


def rectangle_positions(step: int, *, period: int = 120) -> tuple[Array, float, float]:
    """Four corners whose channel-induced cycle crosses an exact double root."""
    modulation = np.sin(2.0 * np.pi * step / period)
    x = 11.5 + modulation
    y = 11.5 - modulation
    positions = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
    return positions, x, y


def simulate_rectangle_crossing(
    rank: int,
    seed: int,
    *,
    total_steps: int,
    consensus_rounds: int,
    channel: MoxChannelParameters,
) -> pd.DataFrame:
    n = 4
    rng = np.random.default_rng(seed)
    node_basis = rng.normal(size=(n, rank))
    rows: list[dict[str, Any]] = []
    for step in range(total_steps):
        positions, x, y = rectangle_positions(step)
        edges, laplacian, mixing, epsilon_c = channel_graph(positions, channel)
        node_basis = distributed_block_step(
            node_basis,
            edges,
            mixing,
            diffusion_step=1.2,
            consensus_rounds=consensus_rounds,
        )
        local_ritz = local_ritz_contributions(node_basis, edges)
        ritz = n * average_consensus(local_ritz, mixing, consensus_rounds)
        ritz = 0.5 * (ritz + np.swapaxes(ritz, -1, -2))
        node_lambda = np.linalg.eigvalsh(ritz)[:, 0]
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        truth = eigenvectors[:, 1 : rank + 1]
        support = active_topology(edges)
        horizontal_weight = next(weight for i, j, weight in edges if (i, j) == (0, 1))
        vertical_weight = next(weight for i, j, weight in edges if (i, j) == (1, 2))
        rows.append(
            {
                "method": "ours-r1-ablation" if rank == 1 else "ours-r2-block",
                "rank": rank,
                "seed": seed,
                "step": step,
                "x_half_width_m": x,
                "y_half_height_m": y,
                "horizontal_weight": horizontal_weight,
                "vertical_weight": vertical_weight,
                "lambda2_true": float(eigenvalues[1]),
                "lambda3_true": float(eigenvalues[2]),
                "fiedler_gap": float(eigenvalues[2] - eigenvalues[1]),
                "lambda2_estimate_mean": float(np.mean(node_lambda)),
                "lambda2_estimate_min": float(np.min(node_lambda)),
                "lambda2_estimate_max": float(np.max(node_lambda)),
                "positive_estimation_error": float(
                    max(0.0, np.mean(node_lambda) - eigenvalues[1])
                ),
                "subspace_sine_error": grassmann_sine_error(node_basis, truth),
                "active_edge_count": len(support),
                "consensus_step": epsilon_c,
                "consensus_contraction": float(
                    np.max(np.abs(1.0 - epsilon_c * eigenvalues[1:]))
                ),
                **{
                    f"component_{i + 1}": float(node_basis[i, 0])
                    for i in range(n)
                },
            }
        )
    return pd.DataFrame(rows)


def _rk4_step(state: Array, laplacian: Array, rhs: Any, dt: float) -> Array:
    """One explicit RK4 step for a frozen communication graph."""
    k1 = rhs(state, laplacian)
    k2 = rhs(state + 0.5 * dt * k1, laplacian)
    k3 = rhs(state + 0.5 * dt * k2, laplacian)
    k4 = rhs(state + dt * k3, laplacian)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _normalized_fiedler_seed(seed: int, n: int) -> Array:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=n)
    vector -= np.mean(vector)
    vector *= np.sqrt(n / np.dot(vector, vector))
    return vector


def _yang_rhs(state: Array, laplacian: Array) -> Array:
    """Yang et al. (Automatica 2010) PI-average-consensus realization."""
    n = len(laplacian)
    x, z1, w1, z2, w2 = np.split(state, 5)
    k1, k2, k3 = 18.0, 3.0, 60.0
    gamma, kp, ki = 100.0, 50.0, 200.0
    x_dot = -k1 * z1 - k2 * (laplacian @ x) - k3 * (z2 - 1.0) * x
    z1_dot = gamma * (x - z1) - kp * (laplacian @ z1) + ki * (laplacian @ w1)
    w1_dot = -ki * (laplacian @ z1)
    x_squared = x * x
    z2_dot = (
        gamma * (x_squared - z2)
        - kp * (laplacian @ z2)
        + ki * (laplacian @ w2)
    )
    w2_dot = -ki * (laplacian @ z2)
    return np.concatenate((x_dot, z1_dot, w1_dot, z2_dot, w2_dot))


def _zhang_high_pass_rhs(state: Array, laplacian: Array) -> Array:
    """High-pass-consensus scalar realization following Zhang et al. (2022)."""
    n = len(laplacian)
    x, q1, q2 = np.split(state, 3)
    k1, k2, k3, beta = 18.0, 3.0, 60.0, 50.0
    y1 = q1 + x
    y2 = q2 + x * x
    x_dot = -k1 * y1 - k2 * (laplacian @ x) - k3 * (y2 - 1.0) * x
    q1_dot = -beta * (laplacian @ y1)
    q2_dot = -beta * (laplacian @ y2)
    return np.concatenate((x_dot, q1_dot, q2_dot))


def simulate_literature_crossing(
    method: str,
    seed: int,
    *,
    total_steps: int,
    channel: MoxChannelParameters,
    physical_dt: float = 0.025,
    integration_substeps: int = 100,
) -> pd.DataFrame:
    """Run a documented continuous-time literature comparator on one path."""
    n = 4
    x0 = _normalized_fiedler_seed(seed, n)
    if method == "yang-2010-single-fiedler":
        state = np.concatenate((x0, x0.copy(), np.zeros(n), x0 * x0, np.zeros(n)))
        rhs = _yang_rhs
    elif method == "zhang-2022-scalar-monitor":
        state = np.concatenate((x0, np.zeros(n), np.zeros(n)))
        rhs = _zhang_high_pass_rhs
    else:
        raise ValueError(method)

    rows: list[dict[str, Any]] = []
    integration_dt = physical_dt / integration_substeps
    for step in range(total_steps):
        positions, x_half_width, y_half_height = rectangle_positions(step)
        edges, laplacian, _, epsilon_c = channel_graph(positions, channel)
        for _ in range(integration_substeps):
            state = _rk4_step(state, laplacian, rhs, integration_dt)
        x_state = state[:n]
        if method == "yang-2010-single-fiedler":
            second_moment = state[3 * n : 4 * n]
            subspace_error = grassmann_sine_error(
                x_state[:, None], np.linalg.eigh(laplacian)[1][:, 1:2]
            )
        else:
            second_moment = state[2 * n :] + x_state * x_state
            # The Zhang comparator is used only through its scalar monitoring
            # output, so no vector/subspace metric is attributed to it.
            subspace_error = np.nan
        node_lambda = 20.0 * (1.0 - second_moment)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        horizontal_weight = next(weight for i, j, weight in edges if (i, j) == (0, 1))
        vertical_weight = next(weight for i, j, weight in edges if (i, j) == (1, 2))
        rows.append(
            {
                "method": method,
                "rank": 1 if method.startswith("yang") else 0,
                "seed": seed,
                "step": step,
                "x_half_width_m": x_half_width,
                "y_half_height_m": y_half_height,
                "horizontal_weight": horizontal_weight,
                "vertical_weight": vertical_weight,
                "lambda2_true": float(eigenvalues[1]),
                "lambda3_true": float(eigenvalues[2]),
                "fiedler_gap": float(eigenvalues[2] - eigenvalues[1]),
                "lambda2_estimate_mean": float(np.mean(node_lambda)),
                "lambda2_estimate_min": float(np.min(node_lambda)),
                "lambda2_estimate_max": float(np.max(node_lambda)),
                "positive_estimation_error": float(
                    max(0.0, np.mean(node_lambda) - eigenvalues[1])
                ),
                "subspace_sine_error": subspace_error,
                "active_edge_count": len(active_topology(edges)),
                "consensus_step": epsilon_c,
                "consensus_contraction": float(
                    np.max(np.abs(1.0 - epsilon_c * eigenvalues[1:]))
                ),
                **{f"component_{i + 1}": float(x_state[i]) for i in range(n)},
            }
        )
    return pd.DataFrame(rows)


def run_rectangle_crossing(
    *,
    total_steps: int = 360,
    seeds: int = 10,
    consensus_rounds: int = 80,
    channel: MoxChannelParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        simulate_rectangle_crossing(
            rank,
            seed,
            total_steps=total_steps,
            consensus_rounds=consensus_rounds,
            channel=channel,
        )
        for rank in (1, 2)
        for seed in range(1, seeds + 1)
    ]
    frames.extend(
        simulate_literature_crossing(
            method,
            seed,
            total_steps=total_steps,
            channel=channel,
        )
        for method in ("yang-2010-single-fiedler", "zhang-2022-scalar-monitor")
        for seed in range(1, seeds + 1)
    )
    data = pd.concat(frames, ignore_index=True)
    burn_in = min(60, total_steps // 6)
    summaries: list[dict[str, Any]] = []
    for (method, seed), subset in data[data["step"] >= burn_in].groupby(
        ["method", "seed"], sort=False
    ):
        error = subset["lambda2_estimate_mean"] - subset["lambda2_true"]
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "lambda2_RMSE": float(np.sqrt(np.mean(error**2))),
                "maximum_positive_overestimate": float(np.max(np.maximum(error, 0.0))),
                "mean_subspace_sine_error": float(subset["subspace_sine_error"].mean()),
                "mean_node_estimate_disagreement": float(
                    (subset["lambda2_estimate_max"] - subset["lambda2_estimate_min"]).mean()
                ),
                "minimum_fiedler_gap": float(subset["fiedler_gap"].min()),
                "minimum_active_edges": int(subset["active_edge_count"].min()),
                "maximum_active_edges": int(subset["active_edge_count"].max()),
                "consensus_rounds": consensus_rounds,
            }
        )
    per_seed = pd.DataFrame(summaries)
    summary = (
        per_seed.groupby("method", sort=False)
        .agg(
            independent_runs=("seed", "count"),
            lambda2_RMSE_mean=("lambda2_RMSE", "mean"),
            lambda2_RMSE_std=("lambda2_RMSE", "std"),
            maximum_positive_overestimate=("maximum_positive_overestimate", "max"),
            mean_subspace_sine_error=("mean_subspace_sine_error", "mean"),
            mean_node_estimate_disagreement=("mean_node_estimate_disagreement", "mean"),
            minimum_fiedler_gap=("minimum_fiedler_gap", "min"),
            minimum_active_edges=("minimum_active_edges", "min"),
            maximum_active_edges=("maximum_active_edges", "max"),
            consensus_rounds=("consensus_rounds", "first"),
        )
        .reset_index()
    )
    return data, summary


SIMPLE_INITIAL = np.array(
    [
        [12.007569, 0.090209],
        [5.862101, 10.392923],
        [-6.029547, 10.608789],
        [-12.061709, -0.052667],
        [-6.115864, -10.153500],
        [6.006322, -10.322051],
    ]
)

SIMPLE_GOAL = np.array(
    [
        [19.972591, 1.046719],
        [10.000000, 17.320508],
        [-9.079810, 17.820130],
        [-19.987817, -0.697990],
        [-10.598385, -16.960962],
        [10.300761, -17.143346],
    ]
)

# Stress-test target used only by the exact-subspace topology-formation study.
# Robot 1 is assigned a farther radial target so that unconstrained nominal
# tracking eventually removes both incident links and isolates that robot.
# The milder SIMPLE_GOAL remains the target of the end-to-end estimator study.
TOPOLOGY_INITIAL = SIMPLE_INITIAL.copy()
TOPOLOGY_GOAL = SIMPLE_GOAL.copy()
TOPOLOGY_GOAL[0] = 36.0 * SIMPLE_GOAL[0] / np.linalg.norm(SIMPLE_GOAL[0])


def connected_component_count(
    n: int, support: set[tuple[int, int]]
) -> int:
    """Return the number of components in an undirected support graph."""
    neighbors = [set() for _ in range(n)]
    for i, j in support:
        neighbors[i].add(j)
        neighbors[j].add(i)
    unseen = set(range(n))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            newly_reached = neighbors[node] & unseen
            unseen.difference_update(newly_reached)
            stack.extend(newly_reached)
    return components


def copy_resource_state(state: ResourceState | None, n: int) -> ResourceState:
    if state is None:
        return initial_resource_state(n)
    return ResourceState(y=state.y.copy(), duals=state.duals.copy())


def simulate_topology_formation(
    method: str,
    *,
    total_steps: int,
    channel: MoxChannelParameters,
    initial_positions: Array | None = None,
    goals_override: Array | None = None,
    dt: float = 0.025,
    nominal_gain: float = 1.5,
    alpha: float = 3.0,
    lambda_des: float = 0.205,
    lambda_req: float = 0.195,
    resource_initial_step: float = 0.001,
    resource_radius: float = 1.0,
    allocation_inner_iterations: int = 10,
    consensus_rounds: int = 1,
) -> tuple[pd.DataFrame, Array]:
    positions = (
        TOPOLOGY_INITIAL.copy()
        if initial_positions is None
        else np.asarray(initial_positions, dtype=float).copy()
    )
    goals = (
        TOPOLOGY_GOAL.copy()
        if goals_override is None
        else np.asarray(goals_override, dtype=float).copy()
    )
    if positions.shape != TOPOLOGY_INITIAL.shape or goals.shape != TOPOLOGY_GOAL.shape:
        raise ValueError("initial positions and goals must both have shape (6, 2)")
    n = len(positions)
    pairs = complete_topology(n)
    state = copy_resource_state(None, n)
    central_dual: Array | None = None
    trajectory = [positions.copy()]
    rows: list[dict[str, Any]] = []
    previous_support: set[tuple[int, int]] | None = None
    cumulative_appearances = 0
    cumulative_disappearances = 0
    allocation_updates = 0

    for step in range(total_steps):
        edges, laplacian, mixing, epsilon_c = channel_graph(positions, channel)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        basis = eigenvectors[:, 1:3]
        model = local_conic_model(
            basis,
            positions,
            pairs,
            sigma=1.0,
            alpha=alpha,
            lambda_des=lambda_des,
            channel_parameters=channel,
        )
        nominal = nominal_gain * (goals - positions)
        master_residual = np.nan
        minimum_local_margin = np.nan
        minimum_inner_local_margin = np.nan
        if method == "nominal":
            controls = nominal
        elif method == "centralized":
            controls, central_dual, _ = centralized_soc_control(
                model, nominal, central_dual
            )
        elif method == "aggregate":
            controls, _, _ = aggregate_parameter_control(
                model, nominal, mixing, consensus_rounds
            )
        elif method == "allocation":
            try:
                inner_margins = []
                for _ in range(allocation_inner_iterations):
                    controls, state, info = violation_free_control(
                        model,
                        nominal,
                        laplacian,
                        state,
                        resource_step=diminishing_resource_step(
                            allocation_updates,
                            initial_step=resource_initial_step,
                            exponent=0.6,
                        ),
                        resource_gain=1.0,
                        resource_radius=resource_radius,
                    )
                    allocation_updates += 1
                    inner_margins.append(info["minimum_local_margin"] / SQRT2)
            except RuntimeError as error:
                raise RuntimeError(
                    f"allocation solve failed at physical sample {step} "
                    f"with {len(active_topology(edges))} active edges"
                ) from error
            master_residual = info["allocation_stationarity_residual"]
            minimum_local_margin = info["minimum_local_margin"] / SQRT2
            minimum_inner_local_margin = float(np.min(inner_margins))
        else:
            raise ValueError(method)

        true_matrix, pre_values, _ = true_barrier_matrix(
            positions,
            controls,
            pairs,
            sigma=1.0,
            alpha=alpha,
            lambda_des=lambda_des,
            channel_parameters=channel,
        )
        exact_margin = float(np.linalg.eigvalsh(true_matrix)[0])
        positions = positions + dt * controls
        trajectory.append(positions.copy())

        post_edges, post_laplacian, _, _ = channel_graph(positions, channel)
        post_values = np.linalg.eigvalsh(post_laplacian)
        support = {tuple(sorted(edge)) for edge in active_topology(post_edges)}
        if previous_support is None:
            appearances = 0
            disappearances = 0
        else:
            appearances = len(support - previous_support)
            disappearances = len(previous_support - support)
        cumulative_appearances += appearances
        cumulative_disappearances += disappearances
        previous_support = support

        row: dict[str, Any] = {
            "method": method,
            "step": step,
            "time_s": (step + 1) * dt,
            "lambda_req": lambda_req,
            "lambda_des": lambda_des,
            "lambda2_pre": float(pre_values[1]),
            "lambda2_post": float(post_values[1]),
            "connectivity_safety_margin": float(post_values[1] - lambda_req),
            "lambda3_post": float(post_values[2]),
            "fiedler_gap_post": float(post_values[2] - post_values[1]),
            "exact_global_matrix_margin": exact_margin,
            "minimum_local_matrix_margin": minimum_local_margin,
            "minimum_inner_local_matrix_margin": minimum_inner_local_margin,
            "master_residual_eta_q": master_residual,
            "allocation_inner_iterations": (
                allocation_inner_iterations if method == "allocation" else np.nan
            ),
            "control_correction": float(np.linalg.norm(controls - nominal)),
            "task_error": float(np.linalg.norm(positions - goals)),
            "active_edge_count": len(support),
            "edge_appearances": appearances,
            "edge_disappearances": disappearances,
            "cumulative_edge_appearances": cumulative_appearances,
            "cumulative_edge_disappearances": cumulative_disappearances,
            "consensus_step": epsilon_c,
            "consensus_contraction": float(
                np.max(np.abs(1.0 - epsilon_c * post_values[1:]))
            ),
            "resource_zero_sum_error": (
                float(np.linalg.norm(np.sum(laplacian @ state.y, axis=0)))
                if method == "allocation"
                else np.nan
            ),
        }
        for i in range(n):
            row[f"p{i + 1}_x_m"] = positions[i, 0]
            row[f"p{i + 1}_y_m"] = positions[i, 1]
        rows.append(row)
    return pd.DataFrame(rows), np.asarray(trajectory)


def run_topology_formation(
    *,
    total_steps: int = 240,
    channel: MoxChannelParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Array]]:
    frames: list[pd.DataFrame] = []
    trajectories: dict[str, Array] = {}
    for method in ("nominal", "centralized", "aggregate", "allocation"):
        frame, trajectory = simulate_topology_formation(
            method,
            total_steps=total_steps,
            channel=channel,
        )
        frames.append(frame)
        trajectories[method] = trajectory
    data = pd.concat(frames, ignore_index=True)
    summaries: list[dict[str, Any]] = []
    for method, subset in data.groupby("method", sort=False):
        terminal_row = subset.iloc[-1]
        terminal_positions = np.asarray(
            [
                [terminal_row[f"p{i + 1}_x_m"], terminal_row[f"p{i + 1}_y_m"]]
                for i in range(6)
            ],
            dtype=float,
        )
        terminal_edges, _, _, _ = channel_graph(terminal_positions, channel)
        terminal_support = {
            tuple(sorted(edge)) for edge in active_topology(terminal_edges)
        }
        summaries.append(
            {
                "method": method,
                "minimum_lambda2": float(subset["lambda2_post"].min()),
                "minimum_connectivity_safety_margin": float(
                    subset["connectivity_safety_margin"].min()
                ),
                "negative_connectivity_safety_samples": int(
                    np.sum(subset["connectivity_safety_margin"] < -1e-9)
                ),
                "minimum_fiedler_gap": float(subset["fiedler_gap_post"].min()),
                "minimum_exact_global_matrix_margin": float(
                    subset["exact_global_matrix_margin"].min()
                ),
                "minimum_inner_local_matrix_margin": (
                    float(subset["minimum_inner_local_matrix_margin"].min())
                    if method == "allocation"
                    else np.nan
                ),
                "negative_exact_margin_samples": int(
                    np.sum(subset["exact_global_matrix_margin"] < -1e-9)
                ),
                "minimum_active_edges": int(subset["active_edge_count"].min()),
                "maximum_active_edges": int(subset["active_edge_count"].max()),
                "total_edge_appearances": int(subset["edge_appearances"].sum()),
                "total_edge_disappearances": int(subset["edge_disappearances"].sum()),
                "final_active_edges": int(subset["active_edge_count"].iloc[-1]),
                "final_lambda2": float(subset["lambda2_post"].iloc[-1]),
                "final_connected_components": connected_component_count(
                    len(terminal_positions), terminal_support
                ),
                "maximum_control_correction": float(
                    subset["control_correction"].max()
                ),
                "final_task_error": float(subset["task_error"].iloc[-1]),
                "allocation_inner_iterations": (
                    int(subset["allocation_inner_iterations"].iloc[-1])
                    if method == "allocation"
                    else np.nan
                ),
                "minimum_master_residual_eta_q": (
                    float(subset["master_residual_eta_q"].min())
                    if method == "allocation"
                    else np.nan
                ),
                "maximum_master_residual_eta_q": (
                    float(subset["master_residual_eta_q"].max())
                    if method == "allocation"
                    else np.nan
                ),
                "maximum_resource_zero_sum_error": (
                    float(subset["resource_zero_sum_error"].max())
                    if method == "allocation"
                    else np.nan
                ),
            }
        )
    return data, pd.DataFrame(summaries), trajectories


def regular_hexagon(radius: float = 17.5) -> Array:
    angles = 2.0 * np.pi * np.arange(6) / 6
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def repeated_root_nominal(positions: Array) -> Array:
    del positions
    # Fixed asymmetric outward task.  The asymmetry leaves one compressed
    # barrier mode strictly positive at the centralized optimum and avoids
    # placing the Lorentz constraint at its numerically degenerate vertex.
    return np.array(
        [
            [1.14061, 0.39658],
            [0.11844, 3.06508],
            [-1.77804, 0.51351],
            [-0.60213, 0.95740],
            [-0.12081, -1.40461],
            [1.04673, -2.58031],
        ]
    )


def run_repeated_root_recovery(
    *,
    iterations: int = 6000,
    channel: MoxChannelParameters,
    lambda_des: float = 0.205,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = regular_hexagon()
    n = len(positions)
    pairs = complete_topology(n)
    edges, laplacian, mixing, epsilon_c = channel_graph(positions, channel)
    basis, eigenvalues, _ = exact_low_basis(
        positions, pairs, sigma=1.0, channel_parameters=channel
    )
    model = local_conic_model(
        basis,
        positions,
        pairs,
        sigma=1.0,
        alpha=3.0,
        lambda_des=lambda_des,
        channel_parameters=channel,
    )
    nominal = repeated_root_nominal(positions)
    gamma_stars = all_gamma_stars(model, angular_samples=1440)
    central, central_dual, central_info = centralized_soc_control(model, nominal)
    central_objective = objective(central, nominal)

    angle = 0.731
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_model = local_conic_model(
        basis @ rotation,
        positions,
        pairs,
        sigma=1.0,
        alpha=3.0,
        lambda_des=lambda_des,
        channel_parameters=channel,
    )
    rotated_central, _, _ = centralized_soc_control(rotated_model, nominal)

    state = initial_resource_state(n)
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations + 1):
        controls, state, info = violation_free_control(
            model,
            nominal,
            laplacian,
            state,
            resource_step=diminishing_resource_step(
                iteration, initial_step=0.0005, exponent=0.6
            ),
            resource_gain=1.0,
            resource_radius=1.0,
        )
        true_matrix, _, _ = true_barrier_matrix(
            positions,
            controls,
            pairs,
            sigma=1.0,
            alpha=3.0,
            lambda_des=lambda_des,
            channel_parameters=channel,
        )
        rows.append(
            {
                "iteration": iteration,
                "lambda2": float(eigenvalues[1]),
                "lambda3": float(eigenvalues[2]),
                "fiedler_gap": float(eigenvalues[2] - eigenvalues[1]),
                "control_error_to_central": float(np.linalg.norm(controls - central)),
                "objective": objective(controls, nominal),
                "objective_gap": objective(controls, nominal) - central_objective,
                "exact_global_matrix_margin": float(np.linalg.eigvalsh(true_matrix)[0]),
                "minimum_local_matrix_margin": info["minimum_local_margin"] / SQRT2,
                "master_residual_eta_q": info["allocation_stationarity_residual"],
                "resource_zero_sum_error": info["resource_zero_sum_error"],
                "resource_projection_count": info["resource_projection_count"],
                "maximum_resource_state_norm": info["maximum_resource_state_norm"],
                "active_edge_count": len(active_topology(edges)),
                "consensus_step": epsilon_c,
                "consensus_contraction": float(
                    np.max(np.abs(1.0 - epsilon_c * eigenvalues[1:]))
                ),
            }
        )
    data = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "lambda2 frozen", "value": float(eigenvalues[1])},
            {"metric": "lambda3 frozen", "value": float(eigenvalues[2])},
            {"metric": "absolute repeated-root gap", "value": float(abs(eigenvalues[2] - eigenvalues[1]))},
            {"metric": "active channel-induced edges", "value": float(len(active_topology(edges)))},
            {"metric": "minimum local Slater diagnostic gamma_i_star", "value": float(np.min(gamma_stars))},
            {"metric": "maximum local Slater diagnostic gamma_i_star", "value": float(np.max(gamma_stars))},
            {"metric": "central objective", "value": central_objective},
            {"metric": "nominal-to-central control correction", "value": float(np.linalg.norm(central - nominal))},
            {"metric": "central exact matrix margin", "value": central_info["global_margin"] / SQRT2},
            {"metric": "rotated-basis centralized-control difference", "value": float(np.linalg.norm(rotated_central - central))},
            {"metric": "initial control error to central", "value": float(data["control_error_to_central"].iloc[0])},
            {"metric": "final control error to central", "value": float(data["control_error_to_central"].iloc[-1])},
            {"metric": "initial objective gap", "value": float(data["objective_gap"].iloc[0])},
            {"metric": "final objective gap", "value": float(data["objective_gap"].iloc[-1])},
            {"metric": "minimum exact matrix margin over all iterates", "value": float(data["exact_global_matrix_margin"].min())},
            {"metric": "final master residual eta_q", "value": float(data["master_residual_eta_q"].iloc[-1])},
            {"metric": "maximum resource zero-sum error", "value": float(data["resource_zero_sum_error"].max())},
            {"metric": "total active resource projections", "value": float(data["resource_projection_count"].sum())},
            {"metric": "maximum resource-state norm", "value": float(data["maximum_resource_state_norm"].max())},
            {"metric": "central dual norm", "value": float(np.linalg.norm(central_dual))},
            {"metric": "consensus step", "value": epsilon_c},
            {"metric": "resource step initial value", "value": 0.0005},
            {"metric": "resource step exponent", "value": 0.6},
            {"metric": "resource projection radius", "value": 1.0},
        ]
    )
    return data, summary


def draw_active_graph(
    axis: plt.Axes,
    positions: Array,
    channel: MoxChannelParameters,
    *,
    title: str,
) -> int:
    edges, _, _, _ = channel_graph(positions, channel)
    positive = [(i, j, w) for i, j, w in edges if w > 0.0]
    maximum = max((w for _, _, w in positive), default=1.0)
    for i, j, weight in positive:
        normalized = weight / maximum
        axis.plot(
            positions[[i, j], 0],
            positions[[i, j], 1],
            color=COLORS["secondary"],
            linewidth=0.5 + 2.0 * normalized,
            alpha=0.25 + 0.7 * normalized,
            zorder=1,
        )
    axis.scatter(positions[:, 0], positions[:, 1], color=COLORS["allocation"], s=22, zorder=2)
    for i, point in enumerate(positions):
        axis.text(point[0] + 0.35, point[1] + 0.35, str(i + 1), fontsize=7)
    axis.set(title=f"{title}: {len(positive)} active edges", xlabel="$p_x$ (m)", ylabel="$p_y$ (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    return len(positive)


def plot_rectangle_crossing(
    data: pd.DataFrame,
    output_dir: Path,
    channel: MoxChannelParameters,
    *,
    version: str,
) -> None:
    selected = data[data["seed"] == 1]
    displayed = selected[selected["step"] >= min(60, int(selected["step"].max()) // 2)]
    block = displayed[displayed["method"] == "ours-r2-block"]
    figure, axes = plt.subplots(2, 2, figsize=(6.9, 4.6))
    axes[0, 0].plot(block["step"], block["lambda2_true"], color=COLORS["truth"], label=r"true $\lambda_2$")
    axes[0, 0].plot(block["step"], block["lambda3_true"], color=COLORS["lambda3"], linestyle="--", label=r"true $\lambda_3$")
    styles = {
        "ours-r2-block": (COLORS["block"], "-", "ours, $r=2$"),
        "ours-r1-ablation": (COLORS["single"], "-.", "ours, $r=1$ ablation"),
        "yang-2010-single-fiedler": (COLORS["yang"], "--", "Yang et al."),
        "zhang-2022-scalar-monitor": (COLORS["zhang"], ":", "Zhang et al."),
    }
    for method, (color, linestyle, label) in styles.items():
        subset = displayed[displayed["method"] == method]
        axes[0, 0].plot(
            subset["step"], subset["lambda2_estimate_mean"],
            color=color, linestyle=linestyle, label="_nolegend_",
        )
        axes[0, 1].plot(
            subset["step"], subset["lambda2_estimate_mean"] - subset["lambda2_true"],
            color=color, linestyle=linestyle, label=label,
        )
    axes[0, 1].axhline(0.0, color=COLORS["truth"], linewidth=0.8)
    axes[1, 0].plot(block["step"], block["horizontal_weight"], color=COLORS["centralized"], label="horizontal")
    axes[1, 0].plot(block["step"], block["vertical_weight"], color=COLORS["aggregate"], linestyle="--", label="vertical")
    axes[1, 1].plot(block["step"], block["fiedler_gap"], color=COLORS["safe"], label=r"$\lambda_3-\lambda_2$")
    axes[1, 1].axhline(0.0, color=COLORS["truth"], linewidth=0.8)
    axes[0, 0].set(xlabel="physical sample", ylabel="eigenvalue", title="(a) Exact crossings")
    axes[0, 1].set(xlabel="physical sample", ylabel="signed estimation error", title="(b) Estimation error")
    axes[1, 0].set(xlabel="physical sample", ylabel="channel weight", title="(c) Geometry-induced weights")
    axes[1, 1].set(xlabel="physical sample", ylabel="eigenvalue gap", title="(d) Exact eigenvalue crossings")
    for axis in axes.flat:
        axis.legend(loc="best")
    save_both(figure, output_dir, f"fig_{version}_rectangle_crossing")

    ablation = selected[selected["method"] == "ours-r1-ablation"].set_index("step")
    snapshot_steps = [165, 180, 195] if int(selected["step"].max()) >= 195 else [45, 60, 75]
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True, sharey=True)
    component_values = ablation[[f"component_{i}" for i in range(1, 5)]].to_numpy()
    color_limit = max(float(np.nanmax(np.abs(component_values))), 1e-12)
    scatter = None
    for axis, step, label in zip(axes, snapshot_steps, ("before", "at", "after")):
        positions, _, _ = rectangle_positions(step)
        edges, _, _, _ = channel_graph(positions, channel)
        maximum_weight = max(weight for _, _, weight in edges)
        for i, j, weight in edges:
            if weight <= 0.0:
                continue
            normalized = weight / maximum_weight
            axis.plot(
                positions[[i, j], 0], positions[[i, j], 1],
                color=COLORS["secondary"], linewidth=0.5 + 3.0 * normalized,
                alpha=0.2 + 0.75 * normalized, zorder=1,
            )
        components = ablation.loc[step, [f"component_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=components, cmap="coolwarm",
            vmin=-color_limit, vmax=color_limit, edgecolors="black", linewidths=0.4,
            s=34, zorder=2,
        )
        for i, point in enumerate(positions):
            axis.text(point[0] + 0.45, point[1] + 0.45, str(i + 1), fontsize=7)
        axis.set(
            title=f"{label} crossing ($k={step}$)", xlabel="$p_x$ (m)",
            ylabel="$p_y$ (m)", aspect="equal",
        )
        axis.grid(False)
    figure.subplots_adjust(left=0.08, right=0.87, bottom=0.18, top=0.86, wspace=0.22)
    if scatter is not None:
        color_axis = figure.add_axes([0.90, 0.20, 0.018, 0.62])
        figure.colorbar(scatter, cax=color_axis, label="$r=1$ component")
    figure.savefig(output_dir / f"fig_{version}_crossing_snapshots.pdf")
    figure.savefig(output_dir / f"fig_{version}_crossing_snapshots.png", dpi=300)
    plt.close(figure)


def plot_topology_formation(
    data: pd.DataFrame,
    trajectories: dict[str, Array],
    output_dir: Path,
    channel: MoxChannelParameters,
    *,
    version: str,
) -> None:
    allocation = data[data["method"] == "allocation"].reset_index(drop=True)
    counts = allocation["active_edge_count"].to_numpy()
    middle_candidates = np.where((counts <= 12) & (counts >= 9))[0]
    middle = int(middle_candidates[0]) if len(middle_candidates) else len(allocation) // 2
    indices = [0, middle + 1, len(trajectories["allocation"]) - 1]
    titles = ["initial", "intermediate", "final"]
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for axis, index, title in zip(axes, indices, titles):
        draw_active_graph(axis, trajectories["allocation"][index], channel, title=title)
    save_both(figure, output_dir, f"fig_{version}_topology_snapshots")

    figure, axes_array = plt.subplots(2, 2, figsize=(6.9, 3.25))
    axes = axes_array.flat
    styles = {
        "nominal": (COLORS["nominal"], "--"),
        "centralized": (COLORS["centralized"], "-."),
        "aggregate": (COLORS["aggregate"], ":"),
        "allocation": (COLORS["allocation"], "-"),
    }
    for method, subset in data[data["method"] != "aggregate"].groupby("method", sort=False):
        color, linestyle = styles[method]
        display_label = "proposed" if method == "allocation" else method
        kwargs = {"color": color, "linestyle": linestyle, "label": display_label}
        axes[0].plot(subset["time_s"], subset["connectivity_safety_margin"], **kwargs)
        axes[1].plot(subset["time_s"], subset["exact_global_matrix_margin"], **kwargs)
        axes[3].plot(subset["time_s"], subset["active_edge_count"], **kwargs)
    axes[0].axhline(0.0, color=COLORS["truth"], linewidth=0.8)
    axes[1].axhline(0.0, color=COLORS["truth"], linewidth=0.8)
    axes[2].semilogy(allocation["time_s"], np.maximum(allocation["master_residual_eta_q"], 1e-14), color=COLORS["allocation"], label=r"$\eta_q$")
    axes[0].set(ylabel=r"$h_{\rm conn}=\lambda_2-\lambda_{\rm req}$", title="(a) Samplewise connectivity safety")
    axes[1].set(ylabel="minimum eigenvalue", title="(b) Exact matrix margin")
    axes[2].set(xlabel="time (s)", ylabel=r"$\eta_q$ (log scale)", title="(c) Nonconverged allocation")
    axes[3].set(xlabel="time (s)", ylabel="positive-weight edges", title="(d) Topology formation")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    axes[2].legend(loc="best")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.84, hspace=0.48, wspace=0.34)
    figure.savefig(output_dir / f"fig_{version}_topology_formation.pdf")
    figure.savefig(output_dir / f"fig_{version}_topology_formation.png", dpi=300)
    plt.close(figure)


def plot_repeated_root(data: pd.DataFrame, output_dir: Path, *, version: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(6.9, 3.25))
    iteration = data["iteration"]
    axes[0, 0].plot(iteration, data["lambda2"], color=COLORS["centralized"], label=r"$\lambda_2$")
    axes[0, 0].plot(iteration, data["lambda3"], color=COLORS["aggregate"], linestyle="--", label=r"$\lambda_3$")
    axes[0, 1].semilogy(iteration, np.maximum(data["control_error_to_central"], 1e-16), color=COLORS["allocation"])
    axes[1, 0].semilogy(iteration, np.maximum(np.abs(data["objective_gap"]), 1e-16), color=COLORS["safe"])
    axes[1, 1].semilogy(iteration, np.maximum(data["exact_global_matrix_margin"], 1e-16), color=COLORS["centralized"], label="matrix margin")
    axes[1, 1].semilogy(iteration, np.maximum(data["master_residual_eta_q"], 1e-16), color=COLORS["allocation"], linestyle="--", label=r"$\eta_q$")
    axes[0, 0].set(ylabel="eigenvalue", title="(a) Channel-induced double root")
    axes[0, 1].set(ylabel=r"$\|u^m-u^\star_{\rm cen}\|_2$", title="(b) Control recovery")
    axes[1, 0].set(xlabel="allocation iteration", ylabel="absolute objective gap", title="(c) Objective recovery")
    axes[1, 1].set(xlabel="allocation iteration", ylabel="value (log scale)", title="(d) Feasibility and stationarity")
    axes[0, 0].legend(loc="best")
    axes[1, 1].legend(loc="best")
    save_both(figure, output_dir, f"fig_{version}_repeated_root_recovery")


def main(*, default_outdir_name: str = "paper_data/core", figure_version: str = "paper") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / default_outdir_name,
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    channel = MoxChannelParameters()

    crossing_steps = 120 if args.quick else 360
    crossing_seeds = 2 if args.quick else 10
    topology_steps = 80 if args.quick else 240
    recovery_iterations = 800 if args.quick else 6000

    crossing, crossing_summary = run_rectangle_crossing(
        total_steps=crossing_steps,
        seeds=crossing_seeds,
        channel=channel,
    )
    topology, topology_summary, trajectories = run_topology_formation(
        total_steps=topology_steps,
        channel=channel,
    )
    repeated, repeated_summary = run_repeated_root_recovery(
        iterations=recovery_iterations,
        channel=channel,
    )

    crossing.to_csv(args.outdir / "rectangle_crossing.csv", index=False)
    crossing_summary.to_csv(args.outdir / "rectangle_crossing_summary.csv", index=False)
    topology.to_csv(args.outdir / "topology_formation.csv", index=False)
    topology_summary.to_csv(args.outdir / "topology_formation_summary.csv", index=False)
    repeated.to_csv(args.outdir / "repeated_root_recovery.csv", index=False)
    repeated_summary.to_csv(args.outdir / "repeated_root_summary.csv", index=False)

    plot_rectangle_crossing(crossing, args.outdir, channel, version=figure_version)
    plot_topology_formation(topology, trajectories, args.outdir, channel, version=figure_version)
    plot_repeated_root(repeated, args.outdir, version=figure_version)

    print("\nRectangle crossing summary:")
    print(crossing_summary.to_string(index=False))
    print("\nTopology formation summary:")
    print(topology_summary.to_string(index=False))
    print("\nRepeated-root summary:")
    print(repeated_summary.to_string(index=False))
    print(f"\nOutputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
