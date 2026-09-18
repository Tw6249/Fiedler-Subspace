"""Network-emulated estimator and local conic safety-filter stack."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from violation_free_soc_core import (
    SQRT2,
    diminishing_resource_step,
    inverse_sqrt_psd,
    j_map,
    lorentz_margin,
    solve_local_soc,
)

from .network_emulator import NeighborPacket, NetworkEmulator
from .scenario import PreflightConfig

Array = np.ndarray


@dataclass
class AgentNode:
    index: int
    basis_row: Array
    allocation_potential: Array
    dual_warm_start: Array

    @classmethod
    def random(cls, index: int, rng: np.random.Generator) -> "AgentNode":
        return cls(
            index=index,
            basis_row=rng.normal(size=2),
            allocation_potential=np.zeros(3),
            dual_warm_start=np.zeros(3),
        )

    def local_ritz_contribution(
        self,
        neighbors: list[NeighborPacket],
    ) -> Array:
        """Return this node's additive compressed-Laplacian contribution."""
        local_ritz = np.zeros((2, 2))
        for packet in neighbors:
            difference = self.basis_row - packet.basis_row
            local_ritz += 0.5 * packet.weight * np.outer(difference, difference)
        return local_ritz

    def local_conic_terms(
        self,
        neighbors: list[NeighborPacket],
        *,
        node_count: int,
        alpha: float,
        lambda_des: float,
        beta: float,
    ) -> tuple[Array, Array, Array]:
        """Build one local model using only received positive-neighbor packets."""
        local_ritz = self.local_ritz_contribution(neighbors)
        control_matrices = np.zeros((2, 2, 2))
        for packet in neighbors:
            difference = self.basis_row - packet.basis_row
            edge_matrix = np.outer(difference, difference)
            for coordinate in range(2):
                control_matrices[coordinate] += (
                    packet.weight_gradient[coordinate] * edge_matrix
                )

        control_map = np.column_stack(
            [j_map(control_matrices[coordinate]) for coordinate in range(2)]
        )
        constant_matrix = alpha * (
            local_ritz - (lambda_des / node_count) * np.eye(2)
        )
        untightened = j_map(constant_matrix)
        tightened = j_map(constant_matrix - beta * np.eye(2))
        return control_map, tightened, untightened


@dataclass(frozen=True)
class ControllerOutput:
    velocity_commands: Array
    basis_rows: Array
    metrics: dict[str, float | int]


class DistributedController:
    """Four neighbor-masked logical nodes connected through NetworkEmulator."""

    def __init__(self, config: PreflightConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        self.agents = [AgentNode.random(i, rng) for i in range(4)]
        self.allocation_updates = 0

    @property
    def basis_rows(self) -> Array:
        return np.vstack([agent.basis_row for agent in self.agents])

    def _set_basis_rows(self, values: Array) -> None:
        for agent, row in zip(self.agents, np.asarray(values, dtype=float)):
            agent.basis_row = row.copy()

    def _new_bus(self, positions: Array) -> NetworkEmulator:
        return NetworkEmulator(
            positions,
            self.config.channel,
            round_latency_ms=self.config.emulated_round_latency_ms,
            message_header_bytes=self.config.message_header_bytes,
            candidate_edges=self.config.candidate_edges,
        )

    def _estimator_step(self, bus: NetworkEmulator) -> dict[str, float]:
        previous = self.basis_rows
        diffused = previous - self.config.diffusion_step * bus.laplacian_apply(previous)
        mean_estimates = bus.consensus(diffused, self.config.consensus_rounds)
        centered = diffused - mean_estimates
        gram_samples = np.einsum("ni,nj->nij", centered, centered)
        gram_estimates = len(previous) * bus.consensus(
            gram_samples,
            self.config.consensus_rounds,
        )
        raw = np.empty_like(previous)
        for index in range(len(previous)):
            raw[index] = centered[index] @ inverse_sqrt_psd(gram_estimates[index])

        cross_samples = np.einsum("ni,nj->nij", raw, previous)
        cross_estimates = len(previous) * bus.consensus(
            cross_samples,
            self.config.consensus_rounds,
        )
        aligned = np.empty_like(previous)
        polar_factors = np.empty((len(previous), 2, 2))
        for index in range(len(previous)):
            left, _, right_t = np.linalg.svd(cross_estimates[index])
            polar_factors[index] = left @ right_t
            aligned[index] = raw[index] @ polar_factors[index]
        self._set_basis_rows(aligned)

        mean_factor = np.mean(polar_factors, axis=0)
        return {
            "minimum_local_gram_eigenvalue": float(
                min(np.linalg.eigvalsh(gram)[0] for gram in gram_estimates)
            ),
            "procrustes_factor_disagreement": float(
                np.max(np.linalg.norm(polar_factors - mean_factor, axis=(1, 2)))
            ),
        }

    def warmup(self, positions: Array) -> None:
        for _ in range(self.config.estimator_warmup_steps):
            self._estimator_step(self._new_bus(positions))

    def _nominal_commands(self, positions: Array, references: Array) -> Array:
        return self.config.nominal_gain * (references - positions)

    def step(
        self,
        positions: Array,
        references: Array,
        *,
        condition: str,
    ) -> ControllerOutput:
        if condition not in {"nominal", "proposed"}:
            raise ValueError("condition must be 'nominal' or 'proposed'")
        started = perf_counter()
        bus = self._new_bus(positions)

        nominal = self._nominal_commands(positions, references)
        if condition == "proposed":
            estimator_started = perf_counter()
            estimator_metrics = self._estimator_step(bus)
            estimator_time_ms = 1e3 * (perf_counter() - estimator_started)
        else:
            estimator_metrics = {
                "minimum_local_gram_eigenvalue": np.nan,
                "procrustes_factor_disagreement": np.nan,
            }
            estimator_time_ms = 0.0

        minimum_inner_local_margin = np.nan
        estimated_tightened_margin = np.nan
        estimated_untightened_margin = np.nan
        allocation_residual = np.nan
        maximum_local_solve_time_ms = 0.0
        total_local_solve_time_ms = 0.0
        parallel_local_solve_time_ms = 0.0
        allocation_time_ms = 0.0
        node_ritz_metrics = {
            "minimum_node_lambda2_estimate": np.nan,
            "maximum_node_lambda2_estimate": np.nan,
            "node_lambda2_estimate_disagreement": np.nan,
            **{
                f"node{index}_lambda2_estimate": np.nan
                for index in range(1, len(self.agents) + 1)
            },
        }
        final_resources: Array | None = None
        local_maps_array: Array | None = None
        constants_array: Array | None = None
        untightened_array: Array | None = None

        if condition == "nominal":
            commands = nominal
        else:
            model_started = perf_counter()
            packets = bus.neighbor_packets(self.basis_rows)
            if self.config.record_node_ritz_estimates:
                local_ritz = np.asarray(
                    [
                        agent.local_ritz_contribution(neighbor_packets)
                        for agent, neighbor_packets in zip(self.agents, packets)
                    ]
                )
                ritz_estimates = len(self.agents) * bus.consensus(
                    local_ritz,
                    self.config.consensus_rounds,
                )
                ritz_estimates = 0.5 * (
                    ritz_estimates + np.swapaxes(ritz_estimates, -1, -2)
                )
                node_lambda2 = np.linalg.eigvalsh(ritz_estimates)[:, 0]
                node_ritz_metrics = {
                    "minimum_node_lambda2_estimate": float(np.min(node_lambda2)),
                    "maximum_node_lambda2_estimate": float(np.max(node_lambda2)),
                    "node_lambda2_estimate_disagreement": float(np.ptp(node_lambda2)),
                    **{
                        f"node{index}_lambda2_estimate": float(value)
                        for index, value in enumerate(node_lambda2, start=1)
                    },
                }
            beta = self.config.beta_total / len(self.agents)
            local_maps = []
            local_constants = []
            local_untightened = []
            for agent, neighbor_packets in zip(self.agents, packets):
                control_map, constant, untightened = agent.local_conic_terms(
                    neighbor_packets,
                    node_count=len(self.agents),
                    alpha=self.config.alpha,
                    lambda_des=self.config.lambda_des,
                    beta=beta,
                )
                local_maps.append(control_map)
                local_constants.append(constant)
                local_untightened.append(untightened)
            local_maps_array = np.asarray(local_maps)
            constants_array = np.asarray(local_constants)
            untightened_array = np.asarray(local_untightened)
            allocation_started = perf_counter()
            minimum_inner_local_margin = np.inf

            for inner_index in range(self.config.allocation_updates):
                potentials = np.vstack(
                    [agent.allocation_potential for agent in self.agents]
                )
                resources = bus.laplacian_apply(potentials)
                final_resources = resources.copy()
                commands = np.empty_like(nominal)
                duals = np.empty((len(self.agents), 3))
                local_margins = np.empty(len(self.agents))
                local_solve_times = np.empty(len(self.agents))
                for index, agent in enumerate(self.agents):
                    local_started = perf_counter()
                    try:
                        solution = solve_local_soc(
                            local_maps_array[index],
                            constants_array[index],
                            resources[index],
                            nominal[index],
                            agent.dual_warm_start,
                        )
                    except RuntimeError as error:
                        raise RuntimeError(
                            "Local SOCP failed for "
                            f"agent {index + 1}, allocation update {inner_index + 1}."
                        ) from error
                    local_solve_times[index] = 1e3 * (
                        perf_counter() - local_started
                    )
                    commands[index] = solution.control
                    duals[index] = solution.dual
                    local_margins[index] = solution.margin / SQRT2
                    agent.dual_warm_start = solution.dual.copy()

                dual_residuals = bus.laplacian_apply(duals)
                step_size = diminishing_resource_step(
                    self.allocation_updates,
                    initial_step=self.config.resource_initial_step,
                    exponent=self.config.resource_step_exponent,
                )
                for index, agent in enumerate(self.agents):
                    updated = agent.allocation_potential - step_size * dual_residuals[index]
                    norm = float(np.linalg.norm(updated))
                    if norm > self.config.resource_radius:
                        updated *= self.config.resource_radius / norm
                    agent.allocation_potential = updated
                self.allocation_updates += 1

                minimum_inner_local_margin = min(
                    minimum_inner_local_margin,
                    float(np.min(local_margins)),
                )
                maximum_local_solve_time_ms = max(
                    maximum_local_solve_time_ms,
                    float(np.max(local_solve_times)),
                )
                total_local_solve_time_ms += float(np.sum(local_solve_times))
                parallel_local_solve_time_ms += float(np.max(local_solve_times))
                allocation_residual = float(np.linalg.norm(dual_residuals))

            allocation_time_ms = 1e3 * (perf_counter() - allocation_started)
            allocation_time_ms += 1e3 * (allocation_started - model_started)

        command_norms = np.linalg.norm(commands, axis=1)
        saturation_mask = command_norms > self.config.command_speed_limit_mps
        if np.any(saturation_mask):
            scales = self.config.command_speed_limit_mps / command_norms[saturation_mask]
            commands[saturation_mask] *= scales[:, None]

        applied_command_norms = np.linalg.norm(commands, axis=1)
        minimum_applied_local_margin = np.nan
        if condition == "proposed":
            assert local_maps_array is not None
            assert constants_array is not None
            assert untightened_array is not None
            assert final_resources is not None
            estimated_tightened_value = np.sum(
                np.einsum("nij,nj->ni", local_maps_array, commands)
                + constants_array,
                axis=0,
            )
            estimated_untightened_value = np.sum(
                np.einsum("nij,nj->ni", local_maps_array, commands)
                + untightened_array,
                axis=0,
            )
            estimated_tightened_margin = (
                lorentz_margin(estimated_tightened_value) / SQRT2
            )
            estimated_untightened_margin = (
                lorentz_margin(estimated_untightened_value) / SQRT2
            )
            applied_local_values = (
                np.einsum("nij,nj->ni", local_maps_array, commands)
                + constants_array
                - final_resources
            )
            minimum_applied_local_margin = float(
                min(lorentz_margin(value) / SQRT2 for value in applied_local_values)
            )

        controller_wall_time_ms = 1e3 * (perf_counter() - started)
        communication = bus.summary()
        virtual_total_time_ms = (
            controller_wall_time_ms
            + float(communication["virtual_communication_time_ms"])
        )
        metrics: dict[str, float | int] = {
            **estimator_metrics,
            **node_ritz_metrics,
            **communication,
            "estimator_time_ms": estimator_time_ms,
            "allocation_time_ms": allocation_time_ms,
            "maximum_local_solve_time_ms": maximum_local_solve_time_ms,
            "total_local_solve_time_ms": total_local_solve_time_ms,
            "parallel_local_solve_time_ms": parallel_local_solve_time_ms,
            "controller_wall_time_ms": controller_wall_time_ms,
            "virtual_total_time_ms": virtual_total_time_ms,
            "deadline_miss": int(
                virtual_total_time_ms > 1e3 * self.config.control_period_s
            ),
            "minimum_inner_local_margin": minimum_inner_local_margin,
            "minimum_applied_local_margin": minimum_applied_local_margin,
            "estimated_tightened_global_margin": estimated_tightened_margin,
            "estimated_untightened_global_margin": estimated_untightened_margin,
            "allocation_stationarity_residual": allocation_residual,
            "maximum_command_speed_mps": float(np.max(command_norms)),
            "maximum_applied_command_speed_mps": float(
                np.max(applied_command_norms)
            ),
            "saturated_node_count": int(np.sum(saturation_mask)),
        }
        return ControllerOutput(
            velocity_commands=commands,
            basis_rows=self.basis_rows.copy(),
            metrics=metrics,
        )
