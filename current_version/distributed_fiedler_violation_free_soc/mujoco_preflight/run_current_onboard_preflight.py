#!/usr/bin/env python3
"""Run the only supported MuJoCo preflight: K=2 and 10 ms per round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from violation_free_soc_core import projector_error, true_barrier_matrix

from .current_onboard_profile import (
    CURRENT_CONSENSUS_ROUNDS,
    CURRENT_ROUTER_ROUND_LATENCY_MS,
    average_communication_rounds_per_second,
    communication_round_breakdown,
    communication_round_service_rate_hz,
    communication_rounds_per_cycle,
)
from .distributed_stack import DistributedController
from .mujoco_plant import MujocoVelocityPlant
from .network_emulator import NetworkEmulator
from .plot_current_onboard_result import plot_current_onboard_result
from .plot_results import plot_results
from .scenario import PreflightConfig, current_preflight_config, rectangle_reference

Array = np.ndarray


def _offline_diagnostics(
    positions: Array,
    velocities: Array,
    estimated_basis: Array,
    config: PreflightConfig,
) -> dict[str, float | int]:
    """Evaluate centralized truth after the distributed command is fixed."""
    diagnostic_graph = NetworkEmulator(
        positions,
        config.channel,
        round_latency_ms=config.emulated_round_latency_ms,
        message_header_bytes=config.message_header_bytes,
        candidate_edges=config.candidate_edges,
    )
    values, _ = np.linalg.eigh(diagnostic_graph.laplacian)
    matrix, _, true_basis = true_barrier_matrix(
        positions,
        velocities,
        config.candidate_edges,
        sigma=1.0,
        alpha=config.alpha,
        lambda_des=config.lambda_des,
        channel_parameters=config.channel,
    )
    return {
        "lambda2": float(values[1]),
        "lambda3": float(values[2]),
        "fiedler_gap": float(values[2] - values[1]),
        "connectivity_margin": float(values[1] - config.lambda_req),
        "exact_barrier_margin": float(np.linalg.eigvalsh(matrix)[0]),
        "subspace_sine_error": projector_error(estimated_basis, true_basis),
        "offline_active_edge_count": len(diagnostic_graph.active_edges),
    }


def _flatten_vectors(row: dict[str, Any], prefix: str, values: Array) -> None:
    for index, value in enumerate(np.asarray(values), start=1):
        row[f"{prefix}{index}_x_m"] = float(value[0])
        row[f"{prefix}{index}_y_m"] = float(value[1])


def _quantized_delay_s(
    projected_time_ms: float,
    period_s: float,
    step_s: float,
) -> float:
    """Round delay upward to a physics step and cap it at one period."""
    if projected_time_ms < 0.0:
        raise ValueError("projected time must be nonnegative")
    delay_steps = int(np.ceil(1e-3 * projected_time_ms / step_s - 1e-12))
    period_steps = int(round(period_s / step_s))
    return min(delay_steps, period_steps) * step_s


def _summarize_current(data: pd.DataFrame, config: PreflightConfig) -> pd.DataFrame:
    summary = {
        "condition": "proposed",
        "architecture_profile": "onboard_via_base_router_delayed",
        "consensus_rounds": config.consensus_rounds,
        "router_round_latency_ms": config.emulated_round_latency_ms,
        "experiment_profile": config.experiment_profile,
        "candidate_topology": config.candidate_topology,
        "samples": len(data),
        "minimum_lambda2": float(data["lambda2"].min()),
        "minimum_connectivity_margin": float(data["connectivity_margin"].min()),
        "below_threshold_samples": int(np.sum(data["connectivity_margin"] < -1e-9)),
        "minimum_fiedler_gap": float(data["fiedler_gap"].min()),
        "minimum_gap_time_s": float(
            data.loc[data["fiedler_gap"].idxmin(), "time_s"]
        ),
        "minimum_exact_command_barrier_margin": float(
            data["exact_command_barrier_margin"].min()
        ),
        "minimum_realized_velocity_barrier_margin": float(
            data["exact_realized_velocity_barrier_margin"].min()
        ),
        "maximum_subspace_sine_error": float(data["subspace_sine_error"].max()),
        "maximum_node_lambda2_absolute_error": float(
            data["maximum_node_lambda2_absolute_error"].max()
        ),
        "maximum_node_lambda2_estimate_disagreement": float(
            data["node_lambda2_estimate_disagreement"].max()
        ),
        "minimum_inner_local_margin": float(data["minimum_inner_local_margin"].min()),
        "minimum_applied_local_margin": float(
            data["minimum_applied_local_margin"].min()
        ),
        "maximum_requested_command_speed_mps": float(
            data["maximum_command_speed_mps"].max()
        ),
        "saturated_node_commands": int(data["saturated_node_count"].sum()),
        "minimum_pair_distance_m": float(data["minimum_pair_distance_m"].min()),
        "minimum_arena_clearance_m": float(
            data["minimum_arena_clearance_m"].min()
        ),
        "mean_controller_wall_time_ms": float(
            data["controller_wall_time_ms"].mean()
        ),
        "maximum_controller_wall_time_ms": float(
            data["controller_wall_time_ms"].max()
        ),
        "maximum_projected_cycle_time_ms": float(data["virtual_total_time_ms"].max()),
        "deadline_misses": int(data["deadline_miss"].sum()),
        "total_communication_rounds": int(data["communication_rounds"].sum()),
        "total_directed_messages": int(data["directed_messages"].sum()),
        "total_communication_bytes": int(data["communication_bytes"].sum()),
        "maximum_node_bytes_per_cycle": int(data["maximum_node_bytes"].max()),
        "control_period_ms": 1e3 * config.control_period_s,
        "consensus_rounds_per_exchange": config.consensus_rounds,
        "allocation_updates_per_cycle": config.allocation_updates,
        "communication_rounds_per_cycle": communication_rounds_per_cycle(),
        "average_communication_rounds_per_second": (
            average_communication_rounds_per_second(
                control_period_s=config.control_period_s
            )
        ),
        "communication_round_service_rate_hz": communication_round_service_rate_hz(),
        "maximum_command_application_delay_ms": float(
            data["command_application_delay_ms"].max()
        ),
        "maximum_stale_position_displacement_m": float(
            data["maximum_stale_position_displacement_m"].max()
        ),
        "late_command_samples": int(
            (data["command_available_within_cycle"] == 0).sum()
        ),
    }
    return pd.DataFrame([summary])


def _run_current_condition(
    config: PreflightConfig,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the current delayed-command condition and save raw evidence."""
    if config.consensus_rounds != CURRENT_CONSENSUS_ROUNDS:
        raise ValueError("current experiment requires K=2")
    if config.emulated_round_latency_ms != CURRENT_ROUTER_ROUND_LATENCY_MS:
        raise ValueError("current experiment requires 10 ms per communication round")

    plant = MujocoVelocityPlant(config)
    controller = DistributedController(config)
    rng = np.random.default_rng(config.random_seed)
    controller.warmup(plant.positions_xy())
    previous_commands = np.zeros((4, 2))

    rows: list[dict[str, Any]] = []
    states_qpos: list[Array] = []
    states_qvel: list[Array] = []
    references_history: list[Array] = []
    for sample in range(config.samples):
        command_time_s = sample * config.control_period_s
        start_positions = plant.positions_xy()
        measured_positions = start_positions + rng.normal(
            0.0,
            config.mocap_noise_std_m,
            size=start_positions.shape,
        )
        references = rectangle_reference(command_time_s, config)
        nominal_commands = config.nominal_gain * (references - measured_positions)
        output = controller.step(measured_positions, references, condition="proposed")
        start_diagnostics = _offline_diagnostics(
            start_positions,
            output.velocity_commands,
            output.basis_rows,
            config,
        )

        projected_time_ms = float(output.metrics["virtual_total_time_ms"])
        delay_s = _quantized_delay_s(
            projected_time_ms,
            config.control_period_s,
            config.physics_timestep_s,
        )
        if delay_s > 0.0:
            plant.hold_velocity(previous_commands, references, delay_s)
        application_positions = plant.positions_xy()
        application_diagnostics = _offline_diagnostics(
            application_positions,
            output.velocity_commands,
            output.basis_rows,
            config,
        )
        remaining_s = config.control_period_s - delay_s
        if remaining_s > 0.0:
            plant.hold_velocity(output.velocity_commands, references, remaining_s)
        previous_commands = output.velocity_commands.copy()

        post_positions = plant.positions_xy()
        post_velocities = plant.velocities_xyz()[:, :2]
        post_diagnostics = _offline_diagnostics(
            post_positions,
            post_velocities,
            output.basis_rows,
            config,
        )
        qpos, qvel = plant.state()
        states_qpos.append(qpos)
        states_qvel.append(qvel)
        references_history.append(references.copy())

        row: dict[str, Any] = {
            "condition": "proposed",
            "sample": sample,
            "command_time_s": command_time_s,
            "time_s": (sample + 1) * config.control_period_s,
            "lambda_req": config.lambda_req,
            "lambda_des": config.lambda_des,
            "lambda2_pre": start_diagnostics["lambda2"],
            "lambda3_pre": start_diagnostics["lambda3"],
            "lambda2": post_diagnostics["lambda2"],
            "lambda3": post_diagnostics["lambda3"],
            "fiedler_gap": post_diagnostics["fiedler_gap"],
            "connectivity_margin": post_diagnostics["connectivity_margin"],
            "exact_command_barrier_margin": application_diagnostics[
                "exact_barrier_margin"
            ],
            "exact_realized_velocity_barrier_margin": post_diagnostics[
                "exact_barrier_margin"
            ],
            "subspace_sine_error": start_diagnostics["subspace_sine_error"],
            "active_edge_count": post_diagnostics["offline_active_edge_count"],
            "minimum_pair_distance_m": plant.minimum_pair_distance(),
            "minimum_arena_clearance_m": float(
                np.min(config.geofence_half_extents_m - np.abs(post_positions))
            ),
            "task_tracking_error_m": float(np.linalg.norm(post_positions - references)),
            "control_correction_norm_mps": float(
                np.linalg.norm(output.velocity_commands - nominal_commands)
            ),
            "velocity_tracking_error_mps": float(
                np.linalg.norm(post_velocities - output.velocity_commands)
            ),
            "router_round_latency_ms": config.emulated_round_latency_ms,
            "command_application_delay_ms": 1e3 * delay_s,
            "command_available_within_cycle": int(
                projected_time_ms <= 1e3 * config.control_period_s
            ),
            "maximum_stale_position_displacement_m": float(
                np.max(np.linalg.norm(application_positions - start_positions, axis=1))
            ),
        }
        row.update(output.metrics)
        node_estimates = np.asarray(
            [
                output.metrics[f"node{index}_lambda2_estimate"]
                for index in range(1, len(post_positions) + 1)
            ],
            dtype=float,
        )
        row["maximum_node_lambda2_absolute_error"] = float(
            np.max(np.abs(node_estimates - start_diagnostics["lambda2"]))
        )
        _flatten_vectors(row, "p", post_positions)
        _flatten_vectors(row, "measured_p", measured_positions)
        _flatten_vectors(row, "ref", references)
        _flatten_vectors(row, "u", output.velocity_commands)
        _flatten_vectors(row, "v", post_velocities)
        rows.append(row)

    timeseries = pd.DataFrame(rows)
    rounds_per_cycle = communication_rounds_per_cycle()
    if not np.all(timeseries["communication_rounds"] == rounds_per_cycle):
        raise RuntimeError("recorded communication rounds disagree with the profile")
    summary = _summarize_current(timeseries, config)

    output_directory.mkdir(parents=True)
    timeseries.to_csv(output_directory / "preflight_timeseries.csv", index=False)
    summary.to_csv(output_directory / "preflight_summary.csv", index=False)
    np.savez_compressed(
        output_directory / "mujoco_state_proposed.npz",
        qpos=np.asarray(states_qpos),
        qvel=np.asarray(states_qvel),
        references=np.asarray(references_history),
        time_s=timeseries["time_s"].to_numpy(),
    )
    metadata = config.to_dict()
    metadata.update(
        {
            "architecture_profile": "onboard_via_base_router_delayed",
            "router_round_latency_ms": config.emulated_round_latency_ms,
            "communication_round_service_rate_hz": (
                communication_round_service_rate_hz()
            ),
            "communication_rounds_per_cycle": rounds_per_cycle,
            "average_communication_rounds_per_second": (
                average_communication_rounds_per_second(
                    control_period_s=config.control_period_s
                )
            ),
            "communication_round_breakdown": communication_round_breakdown(),
            "delay_model": (
                "hold previous command while frozen-state neighbor rounds execute; "
                "round upward to one 5 ms MuJoCo physics step"
            ),
        }
    )
    (output_directory / "preflight_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return timeseries, summary


def run_current_onboard_preflight(output_directory: Path) -> pd.DataFrame:
    """Run the current profile once and generate its two result figures."""
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing experiment directory: {output_directory}"
        )
    config = current_preflight_config()
    timeseries, summary = _run_current_condition(config, output_directory)
    metadata = json.loads(
        (output_directory / "preflight_config.json").read_text(encoding="utf-8")
    )
    plot_results(timeseries, config, output_directory)
    plot_current_onboard_result(timeseries, metadata, output_directory)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_current_reproduction",
    )
    args = parser.parse_args()
    summary = run_current_onboard_preflight(args.outdir)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
