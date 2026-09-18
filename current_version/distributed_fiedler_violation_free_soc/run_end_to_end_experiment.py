#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end empirical estimator--controller experiment for the paper.

The applied controller uses only the row-distributed block estimate.  A
central eigendecomposition is evaluated after the control has been selected
and is used only for offline diagnostics (subspace error, exact barrier
margin, and the true algebraic connectivity).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_paper_experiments import (
    SIMPLE_GOAL,
    SIMPLE_INITIAL,
    channel_graph,
    local_ritz_contributions,
)
from violation_free_soc_core import (
    SQRT2,
    MoxChannelParameters,
    active_topology,
    align_basis,
    average_consensus,
    complete_topology,
    diminishing_resource_step,
    initial_resource_state,
    inverse_sqrt_psd,
    local_conic_model,
    projector_error,
    spectral_model_error_bound,
    true_barrier_matrix,
    violation_free_control,
)

Array = np.ndarray


def distributed_aligned_block_step(
    node_basis: Array,
    edges: list[tuple[int, int, float]],
    mixing: Array,
    *,
    diffusion_step: float,
    consensus_rounds: int,
) -> tuple[Array, dict[str, float]]:
    """Apply one finite-consensus block step and distributed Procrustes alignment."""
    previous = np.asarray(node_basis, dtype=float)
    n = len(previous)
    diffused = previous.copy()
    for i, j, weight in edges:
        difference = previous[i] - previous[j]
        diffused[i] -= diffusion_step * weight * difference
        diffused[j] += diffusion_step * weight * difference

    mean_estimates = average_consensus(diffused, mixing, consensus_rounds)
    centered = diffused - mean_estimates
    gram_samples = np.einsum("ni,nj->nij", centered, centered)
    gram_estimates = n * average_consensus(
        gram_samples,
        mixing,
        consensus_rounds,
    )
    raw = np.empty_like(previous)
    for i in range(n):
        raw[i] = centered[i] @ inverse_sqrt_psd(gram_estimates[i])

    exact_centered = diffused - np.mean(diffused, axis=0, keepdims=True)
    exact_gram = exact_centered.T @ exact_centered
    exact_raw = exact_centered @ inverse_sqrt_psd(exact_gram)
    exact_cross = exact_raw.T @ previous

    cross_samples = np.einsum("ni,nj->nij", raw, previous)
    cross_estimates = n * average_consensus(
        cross_samples,
        mixing,
        consensus_rounds,
    )
    aligned = np.empty_like(raw)
    polar_factors = np.empty((n, 2, 2))
    for i in range(n):
        left, _, right_t = np.linalg.svd(cross_estimates[i])
        polar_factors[i] = left @ right_t
        aligned[i] = raw[i] @ polar_factors[i]

    mean_factor = np.mean(polar_factors, axis=0)
    factor_disagreement = float(
        np.max(np.linalg.norm(polar_factors - mean_factor, axis=(1, 2)))
    )
    diagnostics = {
        "maximum_prenormalized_row_norm": float(
            np.max(np.linalg.norm(centered, axis=1))
        ),
        "minimum_local_gram_eigenvalue": float(
            min(np.linalg.eigvalsh(gram_estimates[i])[0] for i in range(n))
        ),
        "minimum_exact_cross_singular_value": float(
            np.linalg.svd(exact_cross, compute_uv=False)[-1]
        ),
        "procrustes_factor_disagreement": factor_disagreement,
    }
    return aligned, diagnostics


def initialize_estimator(
    positions: Array,
    channel: MoxChannelParameters,
    *,
    seed: int,
    warmup_steps: int,
    diffusion_step: float,
    consensus_rounds: int,
) -> Array:
    """Initialize the row estimator using neighbor operations on the frozen initial graph."""
    rng = np.random.default_rng(seed)
    node_basis = rng.normal(size=(len(positions), 2))
    edges, _, mixing, _ = channel_graph(positions, channel)
    for _ in range(warmup_steps):
        node_basis, _ = distributed_aligned_block_step(
            node_basis,
            edges,
            mixing,
            diffusion_step=diffusion_step,
            consensus_rounds=consensus_rounds,
        )
    return node_basis


def simulate_end_to_end(
    *,
    total_steps: int,
    channel: MoxChannelParameters,
    seed: int = 14,
    warmup_steps: int = 80,
    consensus_rounds: int = 80,
    diffusion_step: float = 0.5,
    dt: float = 0.025,
    nominal_gain: float = 1.5,
    alpha: float = 3.0,
    lambda_des: float = 0.205,
    lambda_req: float = 0.195,
    allocation_inner_iterations: int = 10,
    resource_initial_step: float = 0.001,
    resource_radius: float = 1.0,
    beta_total: float = 0.0,
) -> pd.DataFrame:
    """Run the estimated-subspace controller and return per-sample diagnostics."""
    positions = SIMPLE_INITIAL.copy()
    goals = SIMPLE_GOAL
    n = len(positions)
    pairs = complete_topology(n)
    resource_state = initial_resource_state(n)
    node_basis = initialize_estimator(
        positions,
        channel,
        seed=seed,
        warmup_steps=warmup_steps,
        diffusion_step=diffusion_step,
        consensus_rounds=consensus_rounds,
    )
    allocation_updates = 0
    previous_support: set[tuple[int, int]] | None = None
    rows: list[dict[str, Any]] = []

    for step in range(total_steps):
        edges, laplacian, mixing, epsilon_c = channel_graph(positions, channel)
        node_basis, estimator_diagnostics = distributed_aligned_block_step(
            node_basis,
            edges,
            mixing,
            diffusion_step=diffusion_step,
            consensus_rounds=consensus_rounds,
        )

        # Controller construction below uses the distributed estimate only.
        model = local_conic_model(
            node_basis,
            positions,
            pairs,
            sigma=1.0,
            alpha=alpha,
            lambda_des=lambda_des,
            beta_total=beta_total,
            channel_parameters=channel,
        )
        nominal = nominal_gain * (goals - positions)
        minimum_inner_local_margin = np.inf
        solver_feasible = True
        info: dict[str, float] = {}
        try:
            for _ in range(allocation_inner_iterations):
                controls, resource_state, info = violation_free_control(
                    model,
                    nominal,
                    laplacian,
                    resource_state,
                    resource_step=diminishing_resource_step(
                        allocation_updates,
                        initial_step=resource_initial_step,
                        exponent=0.6,
                    ),
                    resource_gain=1.0,
                    resource_radius=resource_radius,
                )
                allocation_updates += 1
                minimum_inner_local_margin = min(
                    minimum_inner_local_margin,
                    info["minimum_local_margin"] / SQRT2,
                )
        except RuntimeError as error:
            solver_feasible = False
            raise RuntimeError(
                f"Estimated-subspace local solve failed at physical sample {step}."
            ) from error

        # Central spectral information enters only after the control is fixed.
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        true_basis = eigenvectors[:, 1:3]
        true_matrix, _, _ = true_barrier_matrix(
            positions,
            controls,
            pairs,
            sigma=1.0,
            alpha=alpha,
            lambda_des=lambda_des,
            channel_parameters=channel,
        )
        exact_matrix_margin = float(np.linalg.eigvalsh(true_matrix)[0])
        aligned_basis, _ = align_basis(node_basis, true_basis)
        maximum_aligned_row_error = float(
            np.max(np.linalg.norm(aligned_basis - true_basis, axis=1))
        )
        epsilon_x, local_error_bounds, total_error_bound = spectral_model_error_bound(
            node_basis,
            true_basis,
            positions,
            controls,
            pairs,
            sigma=1.0,
            alpha=alpha,
            channel_parameters=channel,
        )
        local_ritz = local_ritz_contributions(node_basis, edges)
        ritz_estimates = n * average_consensus(
            local_ritz,
            mixing,
            consensus_rounds,
        )
        ritz_estimates = 0.5 * (
            ritz_estimates + np.swapaxes(ritz_estimates, -1, -2)
        )
        node_lambda2 = np.linalg.eigvalsh(ritz_estimates)[:, 0]

        positions = positions + dt * controls
        post_edges, post_laplacian, _, _ = channel_graph(positions, channel)
        post_values = np.linalg.eigvalsh(post_laplacian)
        support = {tuple(sorted(edge)) for edge in active_topology(post_edges)}
        if previous_support is None:
            appearances = 0
            disappearances = 0
        else:
            appearances = len(support - previous_support)
            disappearances = len(previous_support - support)
        previous_support = support

        row: dict[str, Any] = {
            "method": "estimated-subspace-empirical",
            "seed": seed,
            "step": step,
            "time_s": (step + 1) * dt,
            "lambda_req": lambda_req,
            "lambda_des": lambda_des,
            "lambda2_pre": float(eigenvalues[1]),
            "lambda2_post": float(post_values[1]),
            "connectivity_safety_margin": float(post_values[1] - lambda_req),
            "lambda3_post": float(post_values[2]),
            "fiedler_gap_post": float(post_values[2] - post_values[1]),
            "lambda2_estimate_mean": float(np.mean(node_lambda2)),
            "lambda2_estimate_max": float(np.max(node_lambda2)),
            "maximum_unsafe_overestimation": float(
                max(0.0, np.max(node_lambda2) - eigenvalues[1])
            ),
            "subspace_sine_error": projector_error(node_basis, true_basis),
            "aligned_basis_spectral_error": epsilon_x,
            "maximum_aligned_row_error": maximum_aligned_row_error,
            "procrustes_factor_disagreement": estimator_diagnostics[
                "procrustes_factor_disagreement"
            ],
            "maximum_prenormalized_row_norm": estimator_diagnostics[
                "maximum_prenormalized_row_norm"
            ],
            "minimum_local_gram_eigenvalue": estimator_diagnostics[
                "minimum_local_gram_eigenvalue"
            ],
            "minimum_exact_cross_singular_value": estimator_diagnostics[
                "minimum_exact_cross_singular_value"
            ],
            "maximum_laplacian_eigenvalue": float(eigenvalues[-1]),
            "consensus_contraction": float(
                np.max(np.abs(1.0 - epsilon_c * eigenvalues[1:]))
            ),
            "exact_global_matrix_margin": exact_matrix_margin,
            "estimated_tightened_global_margin": (
                info["estimated_tightened_global_margin"] / SQRT2
            ),
            "estimated_untightened_global_margin": (
                info["estimated_untightened_global_margin"] / SQRT2
            ),
            "minimum_local_matrix_margin": info["minimum_local_margin"] / SQRT2,
            "minimum_inner_local_matrix_margin": minimum_inner_local_margin,
            "solver_feasible": solver_feasible,
            "offline_total_spectral_error_bound": total_error_bound,
            "offline_maximum_local_error_bound": float(np.max(local_error_bounds)),
            "beta_total": beta_total,
            "master_residual_eta_q": info["allocation_stationarity_residual"],
            "resource_zero_sum_error": info["resource_zero_sum_error"],
            "control_correction": float(np.linalg.norm(controls - nominal)),
            "task_error": float(np.linalg.norm(positions - goals)),
            "active_edge_count": len(support),
            "edge_appearances": appearances,
            "edge_disappearances": disappearances,
            "consensus_rounds": consensus_rounds,
            "consensus_step": epsilon_c,
            "diffusion_step": diffusion_step,
            "warmup_steps": warmup_steps,
            "allocation_inner_iterations": allocation_inner_iterations,
        }
        for i in range(n):
            row[f"p{i + 1}_x_m"] = positions[i, 0]
            row[f"p{i + 1}_y_m"] = positions[i, 1]
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Return the manuscript-facing endpoint, model, and estimator statistics."""
    return pd.DataFrame(
        [
            {
                "method": str(data["method"].iloc[0]),
                "samples": len(data),
                "minimum_lambda2": float(data["lambda2_post"].min()),
                "minimum_connectivity_safety_margin": float(
                    data["connectivity_safety_margin"].min()
                ),
                "negative_connectivity_safety_samples": int(
                    np.sum(data["connectivity_safety_margin"] < -1e-9)
                ),
                "minimum_exact_global_matrix_margin": float(
                    data["exact_global_matrix_margin"].min()
                ),
                "negative_exact_margin_samples": int(
                    np.sum(data["exact_global_matrix_margin"] < -1e-9)
                ),
                "minimum_local_matrix_margin": float(
                    data["minimum_inner_local_matrix_margin"].min()
                ),
                "infeasible_local_solve_samples": int(
                    np.sum(~data["solver_feasible"].astype(bool))
                ),
                "mean_subspace_sine_error": float(
                    data["subspace_sine_error"].mean()
                ),
                "maximum_subspace_sine_error": float(
                    data["subspace_sine_error"].max()
                ),
                "maximum_unsafe_overestimation": float(
                    data["maximum_unsafe_overestimation"].max()
                ),
                "maximum_aligned_row_error": float(
                    data["maximum_aligned_row_error"].max()
                ),
                "maximum_procrustes_factor_disagreement": float(
                    data["procrustes_factor_disagreement"].max()
                ),
                "maximum_prenormalized_row_norm": float(
                    data["maximum_prenormalized_row_norm"].max()
                ),
                "minimum_local_gram_eigenvalue": float(
                    data["minimum_local_gram_eigenvalue"].min()
                ),
                "minimum_exact_cross_singular_value": float(
                    data["minimum_exact_cross_singular_value"].min()
                ),
                "maximum_laplacian_eigenvalue": float(
                    data["maximum_laplacian_eigenvalue"].max()
                ),
                "maximum_consensus_contraction": float(
                    data["consensus_contraction"].max()
                ),
                "maximum_offline_total_spectral_error_bound": float(
                    data["offline_total_spectral_error_bound"].max()
                ),
                "minimum_fiedler_gap": float(data["fiedler_gap_post"].min()),
                "minimum_active_edges": int(data["active_edge_count"].min()),
                "maximum_active_edges": int(data["active_edge_count"].max()),
                "total_edge_appearances": int(data["edge_appearances"].sum()),
                "total_edge_disappearances": int(data["edge_disappearances"].sum()),
                "final_active_edges": int(data["active_edge_count"].iloc[-1]),
                "consensus_rounds": int(data["consensus_rounds"].iloc[0]),
                "warmup_steps": int(data["warmup_steps"].iloc[0]),
                "diffusion_step": float(data["diffusion_step"].iloc[0]),
                "allocation_inner_iterations": int(
                    data["allocation_inner_iterations"].iloc[0]
                ),
                "beta_total": float(data["beta_total"].iloc[0]),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--outdir", type=Path, default=root / "paper_data" / "end_to_end")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--consensus-rounds", type=int, default=80)
    parser.add_argument("--diffusion-step", type=float, default=0.5)
    parser.add_argument("--allocation-iterations", type=int, default=10)
    parser.add_argument("--beta-total", type=float, default=0.0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = simulate_end_to_end(
        total_steps=args.steps,
        channel=MoxChannelParameters(),
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        consensus_rounds=args.consensus_rounds,
        diffusion_step=args.diffusion_step,
        allocation_inner_iterations=args.allocation_iterations,
        beta_total=args.beta_total,
    )
    summary = summarize(data)
    data.to_csv(args.outdir / "end_to_end_estimated.csv", index=False)
    summary.to_csv(args.outdir / "end_to_end_estimated_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
