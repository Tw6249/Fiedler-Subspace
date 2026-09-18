"""Scenario and runtime configuration for the four-UAV preflight campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from violation_free_soc_core import (
    MoxChannelParameters,
    TopologyEdge,
    cycle_topology,
)

Array = np.ndarray


@dataclass(frozen=True)
class PreflightConfig:
    """Parameters shared by the MuJoCo plant and neighbor-masked controller."""

    experiment_profile: str = "current_onboard_k02_100hz"
    candidate_topology: str = "cycle"
    duration_s: float = 40.0
    control_period_s: float = 0.20
    physics_timestep_s: float = 0.005
    altitude_m: float = 1.2
    arena_size_x_m: float = 5.0
    arena_size_y_m: float = 4.0
    arena_boundary_margin_m: float = 0.40
    initial_half_width_m: float = 0.96
    initial_half_height_m: float = 0.56
    crossing_half_extent_m: float = 0.84
    final_half_width_m: float = 0.56
    final_half_height_m: float = 0.96
    final_shear_m: float = 0.0
    crossing_time_s: float = 16.0
    outward_start_s: float = 20.0
    outward_offset_m: float = 0.30
    nominal_gain: float = 2.0
    command_speed_limit_mps: float = 0.50
    alpha: float = 2.0
    lambda_req: float = 0.50
    lambda_des: float = 0.54
    consensus_rounds: int = 2
    estimator_warmup_steps: int = 40
    diffusion_step: float = 0.20
    record_node_ritz_estimates: bool = True
    allocation_updates: int = 1
    resource_initial_step: float = 0.003125
    resource_step_exponent: float = 0.6
    resource_radius: float = 1.0
    beta_total: float = 0.025
    mocap_noise_std_m: float = 0.0016
    emulated_round_latency_ms: float = 10.0
    message_header_bytes: int = 16
    random_seed: int = 31
    inner_velocity_gain: float = 5.0
    altitude_position_gain: float = 18.0
    altitude_velocity_gain: float = 7.0

    def __post_init__(self) -> None:
        if self.experiment_profile != "current_onboard_k02_100hz":
            raise ValueError("only the current onboard experiment profile is supported")
        if self.candidate_topology != "cycle":
            raise ValueError("only the four-cycle candidate topology is supported")
        if self.duration_s <= 0.0 or self.control_period_s <= 0.0:
            raise ValueError("duration and control period must be positive")
        ratio = self.control_period_s / self.physics_timestep_s
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control period must be an integer multiple of the physics timestep")
        if not (0.0 < self.crossing_time_s < self.duration_s):
            raise ValueError("crossing_time_s must lie inside the campaign")
        if not (self.crossing_time_s <= self.outward_start_s < self.duration_s):
            raise ValueError("outward motion must begin after the crossing")
        if self.consensus_rounds < 1 or self.allocation_updates < 1:
            raise ValueError("communication and allocation budgets must be positive")
        if min(self.arena_size_x_m, self.arena_size_y_m) <= 0.0:
            raise ValueError("arena dimensions must be positive")
        if not (
            0.0
            < self.arena_boundary_margin_m
            < 0.5 * min(self.arena_size_x_m, self.arena_size_y_m)
        ):
            raise ValueError("arena boundary margin leaves no usable flight area")
        if np.any(np.abs(self.final_task_targets) > self.geofence_half_extents_m):
            raise ValueError("final task targets must lie inside the arena geofence")

    @property
    def samples(self) -> int:
        return int(round(self.duration_s / self.control_period_s))

    @property
    def initial_positions(self) -> Array:
        return np.array(
            [
                [self.initial_half_width_m, -self.initial_half_height_m],
                [self.initial_half_width_m, self.initial_half_height_m],
                [-self.initial_half_width_m, self.initial_half_height_m],
                [-self.initial_half_width_m, -self.initial_half_height_m],
            ],
            dtype=float,
        )

    @property
    def geofence_half_extents_m(self) -> Array:
        return np.array(
            [
                0.5 * self.arena_size_x_m - self.arena_boundary_margin_m,
                0.5 * self.arena_size_y_m - self.arena_boundary_margin_m,
            ]
        )

    @property
    def candidate_edges(self) -> list[TopologyEdge]:
        return cycle_topology(4)

    @property
    def final_task_targets(self) -> Array:
        references = np.array(
            [
                [self.final_half_width_m, -self.final_half_height_m],
                [self.final_half_width_m, self.final_half_height_m],
                [-self.final_half_width_m, self.final_half_height_m],
                [-self.final_half_width_m, -self.final_half_height_m],
            ],
            dtype=float,
        )
        outward_direction = np.array([1.0, -0.18])
        outward_direction /= np.linalg.norm(outward_direction)
        references[:, 0] += self.final_shear_m * np.array([-1.0, 1.0, 1.0, -1.0])
        references[0] += self.outward_offset_m * outward_direction
        return references

    @property
    def channel(self) -> MoxChannelParameters:
        # The complete 5 m x 4 m task is a horizontal similarity transform of
        # the original task by s=0.8.  Scaling only the transition/cutoff
        # distances would change the Mox core weight.  Because its argument is
        # sqrt(P_T K/P_N0) d^{-n/2}, K must also be scaled by s**n so that the
        # logical graph has the same weight profile at corresponding poses.
        horizontal_scale = 0.8
        path_loss_exponent = 2.52
        return MoxChannelParameters(
            hardware_gain=5.01e-6 * horizontal_scale**path_loss_exponent,
            path_loss_exponent=path_loss_exponent,
            transition_distance=1.60,
            cutoff_distance=2.12,
        )

    def to_dict(self) -> dict[str, float | int | bool | str | list[list[int]]]:
        values = asdict(self)
        values["candidate_edges"] = [list(edge) for edge in self.candidate_edges]
        values["channel_transition_distance_m"] = self.channel.transition_distance
        values["channel_cutoff_distance_m"] = self.channel.cutoff_distance
        values["channel_hardware_gain"] = self.channel.hardware_gain
        values["channel_path_loss_exponent"] = self.channel.path_loss_exponent
        return values


def current_preflight_config() -> PreflightConfig:
    """Return the only supported K=2, 100 Hz-round preflight profile."""
    return PreflightConfig()


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def rectangle_reference(time_s: float, config: PreflightConfig) -> Array:
    """Wide rectangle -> square -> tall rectangle, then pull UAV 1 outward."""
    if time_s <= config.crossing_time_s:
        phase = _smoothstep(time_s / config.crossing_time_s)
        half_width = (
            (1.0 - phase) * config.initial_half_width_m
            + phase * config.crossing_half_extent_m
        )
        half_height = (
            (1.0 - phase) * config.initial_half_height_m
            + phase * config.crossing_half_extent_m
        )
    else:
        phase = _smoothstep(
            (time_s - config.crossing_time_s)
            / (config.duration_s - config.crossing_time_s)
        )
        half_width = (
            (1.0 - phase) * config.crossing_half_extent_m
            + phase * config.final_half_width_m
        )
        half_height = (
            (1.0 - phase) * config.crossing_half_extent_m
            + phase * config.final_half_height_m
        )

    references = np.array(
        [
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
            [-half_width, -half_height],
        ],
        dtype=float,
    )
    if time_s > config.crossing_time_s:
        references[:, 0] += (
            config.final_shear_m
            * phase
            * np.array([-1.0, 1.0, 1.0, -1.0])
        )
    outward_phase = _smoothstep(
        (time_s - config.outward_start_s)
        / (config.duration_s - config.outward_start_s)
    )
    outward_direction = np.array([1.0, -0.18])
    outward_direction /= np.linalg.norm(outward_direction)
    references[0] += config.outward_offset_m * outward_phase * outward_direction
    return references
