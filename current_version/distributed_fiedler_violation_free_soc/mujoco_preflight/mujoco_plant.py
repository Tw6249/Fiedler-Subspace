"""MuJoCo stabilized-UAV velocity plant used by the high-level preflight loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .scenario import PreflightConfig

Array = np.ndarray


class MujocoVelocityPlant:
    """Four second-order translational UAVs behind an autopilot-like velocity loop."""

    def __init__(self, config: PreflightConfig) -> None:
        self.config = config
        model_path = Path(__file__).with_name("scene_four_uav.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        if abs(self.model.opt.timestep - config.physics_timestep_s) > 1e-12:
            raise ValueError("MuJoCo timestep and preflight configuration disagree")

        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"uav{i}")
                for i in range(1, 5)
            ],
            dtype=int,
        )
        self.goal_mocap_ids = np.array(
            [
                self.model.body_mocapid[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"goal{i}",
                    )
                ]
                for i in range(1, 5)
            ],
            dtype=int,
        )
        self.dof_indices = np.empty((4, 3), dtype=int)
        self.actuator_ids = np.empty((4, 3), dtype=int)
        for robot in range(4):
            for coordinate, suffix in enumerate(("x", "y", "z")):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"uav{robot + 1}_{suffix}",
                )
                self.dof_indices[robot, coordinate] = self.model.jnt_dofadr[joint_id]
                self.actuator_ids[robot, coordinate] = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"uav{robot + 1}_f{suffix}",
                )
        self.masses = self.model.body_mass[self.body_ids].copy()
        mujoco.mj_forward(self.model, self.data)
        if not np.allclose(
            self.positions_xy(),
            config.initial_positions,
            atol=1e-12,
        ):
            raise ValueError("MuJoCo XML and configured initial positions disagree")

    def positions_xy(self) -> Array:
        return self.data.xpos[self.body_ids, :2].copy()

    def positions_xyz(self) -> Array:
        return self.data.xpos[self.body_ids].copy()

    def velocities_xyz(self) -> Array:
        return self.data.qvel[self.dof_indices].copy()

    def state(self) -> tuple[Array, Array]:
        """Return a portable MuJoCo generalized-position/velocity snapshot."""
        return self.data.qpos.copy(), self.data.qvel.copy()

    def restore_state(
        self,
        qpos: Array,
        qvel: Array,
        *,
        references: Array | None = None,
    ) -> None:
        """Restore a saved snapshot for deterministic rendering or replay."""
        self.data.qpos[:] = np.asarray(qpos, dtype=float)
        self.data.qvel[:] = np.asarray(qvel, dtype=float)
        if references is not None:
            self.set_goal_markers(references)
        mujoco.mj_forward(self.model, self.data)

    def set_goal_markers(self, references: Array) -> None:
        for index, mocap_id in enumerate(self.goal_mocap_ids):
            self.data.mocap_pos[mocap_id] = np.array(
                [references[index, 0], references[index, 1], 0.025]
            )

    def _set_inner_loop_forces(self, velocity_commands: Array) -> None:
        positions = self.positions_xyz()
        velocities = self.velocities_xyz()
        for index in range(4):
            mass = float(self.masses[index])
            horizontal = (
                mass
                * self.config.inner_velocity_gain
                * (velocity_commands[index] - velocities[index, :2])
            )
            vertical = mass * (
                9.81
                + self.config.altitude_position_gain
                * (self.config.altitude_m - positions[index, 2])
                - self.config.altitude_velocity_gain * velocities[index, 2]
            )
            force = np.array([horizontal[0], horizontal[1], vertical])
            for coordinate in range(3):
                actuator = self.actuator_ids[index, coordinate]
                lower, upper = self.model.actuator_ctrlrange[actuator]
                self.data.ctrl[actuator] = float(
                    np.clip(force[coordinate], lower, upper)
                )

    def hold_velocity(
        self,
        velocity_commands: Array,
        references: Array,
        duration_s: float,
        *,
        viewer: Any | None = None,
    ) -> None:
        steps = int(round(duration_s / self.model.opt.timestep))
        self.set_goal_markers(references)
        for _ in range(steps):
            self._set_inner_loop_forces(velocity_commands)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    def minimum_pair_distance(self) -> float:
        positions = self.positions_xy()
        distances = [
            np.linalg.norm(positions[i] - positions[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        return float(np.min(distances))

    def render(
        self,
        *,
        camera: str = "perspective",
        width: int = 960,
        height: int = 720,
    ) -> Array:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()
