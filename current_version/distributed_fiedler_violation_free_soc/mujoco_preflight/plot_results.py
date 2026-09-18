#!/usr/bin/env python3
"""Create the compact four-panel preflight evidence figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mujoco_preflight.scenario import PreflightConfig
else:
    from .scenario import PreflightConfig


def _condition(data: pd.DataFrame, name: str) -> pd.DataFrame | None:
    selected = data.loc[data["condition"] == name].copy()
    return selected if len(selected) else None


def _positions(data: pd.DataFrame, prefix: str = "p") -> np.ndarray:
    return np.stack(
        [
            data[[f"{prefix}{index}_x_m", f"{prefix}{index}_y_m"]].to_numpy()
            for index in range(1, 5)
        ],
        axis=1,
    )


def _trajectory_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    config: PreflightConfig,
) -> None:
    proposed = _condition(data, "proposed")
    nominal = _condition(data, "nominal")
    displayed = proposed if proposed is not None else nominal
    assert displayed is not None
    positions = _positions(displayed)
    references = _positions(displayed, "ref")
    colors = plt.get_cmap("tab10").colors[:4]

    for robot, color in enumerate(colors):
        robot_positions = positions[:, robot]
        axis.plot(
            robot_positions[:, 0],
            robot_positions[:, 1],
            color=color,
            linewidth=1.6,
        )
        axis.scatter(
            robot_positions[0, 0],
            robot_positions[0, 1],
            s=58,
            facecolors="white",
            edgecolors=[color],
            linewidths=1.4,
            zorder=4,
        )
        axis.text(
            robot_positions[0, 0],
            robot_positions[0, 1],
            str(robot + 1),
            color=color,
            fontsize=6,
            ha="center",
            va="center",
            zorder=5,
        )
        axis.scatter(
            robot_positions[-1, 0],
            robot_positions[-1, 1],
            s=58,
            marker="o",
            color=[color],
            zorder=4,
        )
        axis.text(
            robot_positions[-1, 0],
            robot_positions[-1, 1],
            str(robot + 1),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            zorder=5,
        )

        increments = np.linalg.norm(np.diff(robot_positions, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        if cumulative[-1] > 1e-9:
            arrow_end = int(np.searchsorted(cumulative, 0.62 * cumulative[-1]))
            arrow_start = max(0, arrow_end - 5)
            axis.annotate(
                "",
                xy=robot_positions[arrow_end],
                xytext=robot_positions[arrow_start],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.1,
                    "mutation_scale": 8,
                },
                zorder=3,
            )

    if nominal is not None and proposed is not None:
        nominal_positions = _positions(nominal)
        axis.plot(
            nominal_positions[:, 0, 0],
            nominal_positions[:, 0, 1],
            color="0.25",
            linewidth=1.1,
            linestyle="--",
        )

    task_targets = references[-1]
    for robot, color in enumerate(colors):
        axis.scatter(
            task_targets[robot, 0],
            task_targets[robot, 1],
            s=46 if robot == 0 else 32,
            marker="x",
            color=[color],
            linewidths=1.5 if robot == 0 else 1.1,
            alpha=1.0 if robot == 0 else 0.72,
            zorder=5,
        )
    axis.annotate(
        "$g_1$: outward target",
        xy=task_targets[0],
        xytext=(-64, -14),
        textcoords="offset points",
        fontsize=7,
        color="0.25",
    )
    axis.text(
        0.43,
        0.55,
        "open circles: initial\nfilled circles: final\ncrosses: task targets\ngray dashed: nominal UAV 1",
        transform=axis.transAxes,
        fontsize=7,
        color="0.25",
        ha="center",
        va="center",
    )

    arena_width = config.arena_size_x_m
    arena_height = config.arena_size_y_m
    geofence_width, geofence_height = 2.0 * config.geofence_half_extents_m
    axis.add_patch(
        Rectangle(
            (-0.5 * arena_width, -0.5 * arena_height),
            arena_width,
            arena_height,
            fill=False,
            edgecolor="0.30",
            linewidth=0.9,
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (-0.5 * geofence_width, -0.5 * geofence_height),
            geofence_width,
            geofence_height,
            fill=False,
            edgecolor="0.60",
            linewidth=0.8,
            linestyle=":",
            zorder=0,
        )
    )
    axis.text(
        0.02,
        0.97,
        "solid: 5 m $\\times$ 4 m arena\ndotted: 0.40 m geofence margin",
        transform=axis.transAxes,
        fontsize=6.5,
        color="0.35",
        ha="left",
        va="top",
    )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("$x$ (m)")
    axis.set_ylabel("$y$ (m)")
    axis.set_xlim(-0.5 * arena_width - 0.05, 0.5 * arena_width + 0.05)
    axis.set_ylim(-0.5 * arena_height - 0.05, 0.5 * arena_height + 0.05)
    axis.set_title("(a) Task and trajectory endpoints", loc="left")
    axis.grid(alpha=0.18, linewidth=0.5)


def _spectrum_panel(axis: plt.Axes, data: pd.DataFrame, config: PreflightConfig) -> None:
    proposed = _condition(data, "proposed")
    nominal = _condition(data, "nominal")
    spectral = proposed if proposed is not None else nominal
    assert spectral is not None
    if nominal is not None:
        axis.plot(
            nominal["time_s"],
            nominal["lambda2"],
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            label="Nominal $\\lambda_2$",
        )
    axis.plot(
        spectral["time_s"],
        spectral["lambda2"],
        color="#0072B2",
        linewidth=1.6,
        label=("Proposed $\\lambda_2$" if proposed is not None else "$\\lambda_2$"),
    )
    axis.plot(
        spectral["time_s"],
        spectral["lambda3"],
        color="#D55E00",
        linewidth=1.2,
        label=("Proposed $\\lambda_3$" if proposed is not None else "$\\lambda_3$"),
    )
    if proposed is not None:
        estimate_columns = [
            f"node{index}_lambda2_estimate" for index in range(1, 5)
        ]
        if all(column in proposed for column in estimate_columns):
            node_estimates = proposed[estimate_columns].to_numpy(dtype=float)
            if np.all(np.isfinite(node_estimates)):
                axis.fill_between(
                    proposed["time_s"],
                    np.min(node_estimates, axis=1),
                    np.max(node_estimates, axis=1),
                    color="#56B4E9",
                    alpha=0.18,
                    linewidth=0.0,
                    label="Node Ritz estimate range",
                )
                axis.plot(
                    proposed["time_s"],
                    np.mean(node_estimates, axis=1),
                    color="#0072B2",
                    linewidth=0.9,
                    linestyle="-.",
                    label="Mean node Ritz estimate",
                )
    axis.axhline(
        config.lambda_req,
        color="0.1",
        linewidth=0.9,
        linestyle=":",
        label="$\\lambda_{\\rm req}$",
    )
    crossing_row = spectral.loc[spectral["fiedler_gap"].idxmin()]
    axis.axvline(
        crossing_row["time_s"],
        color="0.65",
        linewidth=0.8,
        zorder=0,
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Laplacian eigenvalue")
    axis.set_title("(b) Spectrum and safety threshold", loc="left")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(alpha=0.18, linewidth=0.5)


def _diagnostic_panel(axis: plt.Axes, data: pd.DataFrame) -> None:
    proposed = _condition(data, "proposed")
    displayed = proposed if proposed is not None else data.copy()
    if proposed is not None:
        axis.plot(
            displayed["time_s"],
            displayed["subspace_sine_error"],
            color="#009E73",
            linewidth=1.4,
            label="Subspace sine error",
        )
        axis.set_ylabel("Subspace sine error", color="#009E73")
        axis.tick_params(axis="y", colors="#009E73")
        margin_axis = axis.twinx()
    else:
        margin_axis = axis
    margin_axis.plot(
        displayed["time_s"],
        displayed["exact_command_barrier_margin"],
        color="#CC79A7",
        linewidth=1.3,
        label="Exact command barrier",
    )
    margin_axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
    margin_axis.set_ylabel("Exact barrier margin", color="#CC79A7")
    margin_axis.tick_params(axis="y", colors="#CC79A7")
    axis.set_xlabel("Time (s)")
    axis.set_title("(c) Subspace and barrier margins", loc="left")
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = margin_axis.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, fontsize=7)
    axis.grid(alpha=0.18, linewidth=0.5)


def _timing_panel(axis: plt.Axes, data: pd.DataFrame, config: PreflightConfig) -> None:
    proposed = _condition(data, "proposed")
    nominal = _condition(data, "nominal")
    displayed = proposed if proposed is not None else nominal
    assert displayed is not None
    axis.plot(
        displayed["time_s"],
        displayed["controller_wall_time_ms"],
        color="#56B4E9",
        linewidth=1.0,
        label="Measured wall time",
    )
    axis.plot(
        displayed["time_s"],
        displayed["virtual_total_time_ms"],
        color="#E69F00",
        linewidth=1.2,
        label="Wall + emulated transport",
    )
    axis.axhline(
        1e3 * config.control_period_s,
        color="0.15",
        linestyle=":",
        linewidth=0.9,
        label="Control deadline",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Execution time (ms)")
    axis.set_title("(d) Cycle timing", loc="left")
    axis.legend(frameon=False, fontsize=7)
    axis.grid(alpha=0.18, linewidth=0.5)


def plot_results(
    data: pd.DataFrame,
    config: PreflightConfig,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Save PDF and PNG versions of the four-panel campaign figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.55), constrained_layout=True)
    _trajectory_panel(axes[0, 0], data, config)
    _spectrum_panel(axes[0, 1], data, config)
    _diagnostic_panel(axes[1, 0], data)
    _timing_panel(axes[1, 1], data, config)
    png_path = output_directory / "fig_mujoco_preflight.png"
    pdf_path = output_directory / "fig_mujoco_preflight.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.output_directory / "preflight_timeseries.csv")
    with (args.output_directory / "preflight_config.json").open(
        encoding="utf-8"
    ) as file:
        raw_config = json.load(file)
    known = PreflightConfig.__dataclass_fields__
    config = PreflightConfig(
        **{key: value for key, value in raw_config.items() if key in known}
    )
    plot_results(data, config, args.output_directory)


if __name__ == "__main__":
    main()
