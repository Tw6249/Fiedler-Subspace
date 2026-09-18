#!/usr/bin/env python3
"""Generate publication figures and metrics for the four-robot experiment.

The script reads the retained physical-run CSV, reconstructs centralized
ground-truth spectral quantities over all six robot pairs only for a posteriori
evaluation, and exports IEEE-ready PDF/PNG figures plus a hash-traceable metrics
record. The applied controller does not receive the centralized
eigendecomposition used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from mujoco_preflight.scenario import current_preflight_config
from violation_free_soc_core import exact_low_basis, true_barrier_matrix


Array = np.ndarray

INK = "#20252A"
MUTED = "#65717C"
GRID = "#D9DEE2"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#D55E00"
GOLD = "#E69F00"
PINK = "#CC79A7"
GRAY = "#8A8A8A"
PALE_BLUE = "#E8F2F7"
PALE_GREEN = "#E7F3ED"
PALE_GOLD = "#FCF2D8"
PALE_RED = "#F8E5E7"

ROBOT_COLORS = (BLUE, ORANGE, GREEN, PINK)
PAIR_LIST = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
# Evaluate all unordered robot pairs. The range model, rather than a manually
# masked cycle, determines which pairs have positive weight.
EVALUATION_EDGES = PAIR_LIST


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.7,
        "axes.linewidth": 0.60,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.48,
        "grid.linewidth": 0.32,
        "lines.linewidth": 1.05,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def limit_rows(vectors: Array, limit: float) -> Array:
    result = np.asarray(vectors, dtype=float).copy()
    norms = np.linalg.norm(result, axis=1)
    mask = norms > limit
    if np.any(mask):
        result[mask] *= (limit / norms[mask])[:, None]
    return result


def load_run(csv_path: Path) -> dict[str, Array | pd.DataFrame]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"physical-run CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required_columns = {
        "elapsed_s",
        "minimum_separation_m",
        "geofence_margin_m",
        "maximum_local_solve_time_ms",
        "controller_wall_time_ms",
        "virtual_total_time_ms",
        "deadline_miss",
        "node_lambda2_estimate_disagreement",
    }
    for index in range(1, 5):
        required_columns.update(
            {
                f"car_{index}_local_x_m",
                f"car_{index}_local_y_m",
                f"car_{index}_reference_x_m",
                f"car_{index}_reference_y_m",
                f"car_{index}_world_cmd_x_mps",
                f"car_{index}_world_cmd_y_mps",
                f"car_{index}_mocap_age_s",
                f"node{index}_lambda2_estimate",
            }
        )
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"physical-run CSV is missing required columns: {missing}")
    time_s = frame["elapsed_s"].to_numpy(float)
    positions = np.stack(
        [
            frame[[f"car_{index}_local_x_m", f"car_{index}_local_y_m"]].to_numpy(float)
            for index in range(1, 5)
        ],
        axis=1,
    )
    references = np.stack(
        [
            frame[
                [f"car_{index}_reference_x_m", f"car_{index}_reference_y_m"]
            ].to_numpy(float)
            for index in range(1, 5)
        ],
        axis=1,
    )
    commands = np.stack(
        [
            frame[
                [f"car_{index}_world_cmd_x_mps", f"car_{index}_world_cmd_y_mps"]
            ].to_numpy(float)
            for index in range(1, 5)
        ],
        axis=1,
    )
    return {
        "frame": frame,
        "time_s": time_s,
        "positions": positions,
        "references": references,
        "commands": commands,
    }


def reconstruct(run: dict[str, Array | pd.DataFrame]) -> dict[str, Array]:
    config = current_preflight_config()
    time_s = np.asarray(run["time_s"])
    positions = np.asarray(run["positions"])
    references = np.asarray(run["references"])
    commands = np.asarray(run["commands"])
    nominal = config.nominal_gain * (references - positions)
    nominal = np.stack(
        [limit_rows(sample, config.command_speed_limit_mps) for sample in nominal]
    )

    lambda_actual = np.empty(len(time_s))
    lambda_reference = np.empty(len(time_s))
    fiedler_gap = np.empty(len(time_s))
    barrier_actual = np.empty(len(time_s))
    barrier_nominal = np.empty(len(time_s))
    pair_distances = np.empty((len(time_s), len(PAIR_LIST)))

    for sample in range(len(time_s)):
        _, actual_values, _ = exact_low_basis(
            positions[sample],
            EVALUATION_EDGES,
            sigma=1.0,
            channel_parameters=config.channel,
        )
        _, reference_values, _ = exact_low_basis(
            references[sample],
            EVALUATION_EDGES,
            sigma=1.0,
            channel_parameters=config.channel,
        )
        actual_matrix, _, _ = true_barrier_matrix(
            positions[sample],
            commands[sample],
            EVALUATION_EDGES,
            sigma=1.0,
            alpha=config.alpha,
            lambda_des=config.lambda_des,
            channel_parameters=config.channel,
        )
        nominal_matrix, _, _ = true_barrier_matrix(
            positions[sample],
            nominal[sample],
            EVALUATION_EDGES,
            sigma=1.0,
            alpha=config.alpha,
            lambda_des=config.lambda_des,
            channel_parameters=config.channel,
        )
        lambda_actual[sample] = actual_values[1]
        lambda_reference[sample] = reference_values[1]
        fiedler_gap[sample] = actual_values[2] - actual_values[1]
        barrier_actual[sample] = np.linalg.eigvalsh(actual_matrix)[0]
        barrier_nominal[sample] = np.linalg.eigvalsh(nominal_matrix)[0]
        for pair_index, (i, j) in enumerate(PAIR_LIST):
            pair_distances[sample, pair_index] = np.linalg.norm(
                positions[sample, i] - positions[sample, j]
            )

    tracking = np.linalg.norm(positions - references, axis=2)
    correction = np.linalg.norm(commands - nominal, axis=2)
    command_speed = np.linalg.norm(commands, axis=2)
    return {
        "nominal": nominal,
        "lambda_actual": lambda_actual,
        "lambda_reference": lambda_reference,
        "fiedler_gap": fiedler_gap,
        "barrier_actual": barrier_actual,
        "barrier_nominal": barrier_nominal,
        "pair_distances": pair_distances,
        "tracking": tracking,
        "correction": correction,
        "command_speed": command_speed,
    }


def contiguous_intervals(time_s: Array, mask: Array) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            left = time_s[start] if start == 0 else 0.5 * (time_s[start - 1] + time_s[start])
            right = time_s[end] if end == len(mask) - 1 else 0.5 * (time_s[end] + time_s[end + 1])
            intervals.append((float(left), float(right)))
            start = None
    return intervals


def classify_pair_histories(
    pair_distances: Array, cutoff_distance: float
) -> tuple[Array, Array, Array]:
    """Return indices of always-admitted, always-rejected, and switching pairs."""
    maximum = np.max(pair_distances, axis=0)
    minimum = np.min(pair_distances, axis=0)
    always_admitted = np.flatnonzero(maximum < cutoff_distance)
    always_rejected = np.flatnonzero(minimum >= cutoff_distance)
    switching = np.flatnonzero(
        (minimum < cutoff_distance) & (maximum >= cutoff_distance)
    )
    return always_admitted, always_rejected, switching


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.13,
        1.05,
        rf"$\bf{{({letter})}}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=INK,
        clip_on=False,
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(figure)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_relative(path: Path) -> str:
    workspace = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return str(path.resolve())


def architecture_node(axis: plt.Axes, x: float, y: float, index: int) -> None:
    box = FancyBboxPatch(
        (x - 0.13, y - 0.09),
        0.26,
        0.18,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=0.75,
        edgecolor=BLUE,
        facecolor=PALE_BLUE,
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(x, y + 0.035, f"Robot {index}", ha="center", va="center", fontsize=8.1, color=INK, zorder=5)
    axis.text(x, y - 0.035, "range filter + local SOCP", ha="center", va="center", fontsize=6.8, color=MUTED, zorder=5)


def plot_architecture(output_dir: Path) -> None:
    """Export the architecture as one panel for LaTeX-side composition."""
    # Match the panel's approximate final aspect and scale so text remains
    # readable when LaTeX places it at half of the two-column text width.
    figure, diagram_axis = plt.subplots(figsize=(4.6, 2.6))
    diagram_axis.set_xlim(0, 1)
    diagram_axis.set_ylim(0, 1)
    diagram_axis.set_axis_off()

    mocap = FancyBboxPatch((0.31, 0.84), 0.38, 0.10, boxstyle="round,pad=0.012", linewidth=0.75, edgecolor=INK, facecolor=PALE_GREEN)
    diagram_axis.add_patch(mocap)
    diagram_axis.text(0.50, 0.89, "Mocap / VRPN localization", ha="center", va="center", fontsize=8.1, color=INK)

    coordinates = ((0.25, 0.63), (0.75, 0.63), (0.75, 0.29), (0.25, 0.29))
    for index, (x, y) in enumerate(coordinates, start=1):
        architecture_node(diagram_axis, x, y, index)
        diagram_axis.annotate("", xy=(x, y + 0.10), xytext=(0.50, 0.84), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.65, mutation_scale=7), zorder=1)

    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        diagram_axis.plot(
            [coordinates[i][0], coordinates[j][0]],
            [coordinates[i][1], coordinates[j][1]],
            color=BLUE,
            linewidth=1.6,
            zorder=2,
        )
    for i, j in ((0, 2), (1, 3)):
        diagram_axis.plot(
            [coordinates[i][0], coordinates[j][0]],
            [coordinates[i][1], coordinates[j][1]],
            color=ORANGE,
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            alpha=0.75,
            zorder=1,
        )

    diagram_axis.text(0.50, 0.49, r"neighbor exchange if $d_{ij}<d_c$", ha="center", va="center", fontsize=7.2, color=BLUE, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9), zorder=6)
    diagram_axis.text(0.50, 0.08, "ROS 2 / Wi-Fi; optimization solved onboard in parallel", ha="center", va="center", fontsize=7.4, color=INK)
    diagram_axis.plot([], [], color=BLUE, linewidth=1.6, label="active neighbor link")
    diagram_axis.plot([], [], color=ORANGE, linewidth=0.9, linestyle=(0, (3, 2)), label="rejected out-of-range pair")
    diagram_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=2, frameon=False, handlelength=2.1, columnspacing=0.9)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save_figure(figure, output_dir, "physical_experiment_architecture")


def plot_results(
    run: dict[str, Array | pd.DataFrame],
    diagnostics: dict[str, Array],
    output_dir: Path,
) -> None:
    config = current_preflight_config()
    time_s = np.asarray(run["time_s"])
    positions = np.asarray(run["positions"])
    references = np.asarray(run["references"])
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.20))
    trajectory_axis, lambda_axis, barrier_axis, distance_axis = axes.ravel()

    for robot in range(4):
        trajectory_axis.plot(
            references[:, robot, 0],
            references[:, robot, 1],
            color=ROBOT_COLORS[robot],
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            alpha=0.62,
        )
        trajectory_axis.plot(
            positions[:, robot, 0],
            positions[:, robot, 1],
            color=ROBOT_COLORS[robot],
            linewidth=1.25,
            label=f"robot {robot + 1}",
        )
        trajectory_axis.plot(positions[0, robot, 0], positions[0, robot, 1], marker="o", markersize=4.2, markerfacecolor="white", markeredgecolor=ROBOT_COLORS[robot], markeredgewidth=0.9)
        trajectory_axis.plot(positions[-1, robot, 0], positions[-1, robot, 1], marker="s", markersize=4.0, color=ROBOT_COLORS[robot])
        trajectory_axis.plot(references[-1, robot, 0], references[-1, robot, 1], marker="x", markersize=4.4, color=ROBOT_COLORS[robot], markeredgewidth=0.9)
    distances = diagnostics["pair_distances"]
    for pair_index, (i, j) in enumerate(PAIR_LIST):
        if distances[-1, pair_index] < config.channel.cutoff_distance:
            trajectory_axis.plot(
                [positions[-1, i, 0], positions[-1, j, 0]],
                [positions[-1, i, 1], positions[-1, j, 1]],
                color=MUTED,
                linewidth=0.7,
                alpha=0.65,
                zorder=0,
            )
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.set_xlabel(r"local $x$ (m)")
    trajectory_axis.set_ylabel(r"local $y$ (m)")
    trajectory_axis.legend(loc="upper center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.7)
    panel_label(trajectory_axis, "a")

    lambda_actual = diagnostics["lambda_actual"]
    lambda_reference = diagnostics["lambda_reference"]
    for left, right in contiguous_intervals(time_s, lambda_reference < config.lambda_req):
        lambda_axis.axvspan(left, right, color=PALE_RED, zorder=0)
    lambda_axis.plot(time_s, lambda_actual, color=BLUE, label=r"measured-state $\lambda_2$")
    lambda_axis.plot(time_s, lambda_reference, color=ORANGE, linestyle=(0, (4, 2)), label=r"reference-state $\lambda_2$")
    lambda_axis.axhline(config.lambda_req, color=INK, linestyle=(0, (2, 2)), linewidth=0.9, label=r"$\lambda_{\rm req}=0.50$")
    minimum_index = int(np.argmin(lambda_actual))
    lambda_axis.plot(time_s[minimum_index], lambda_actual[minimum_index], marker="o", markersize=4.2, color=BLUE)
    lambda_axis.annotate(
        rf"min. {lambda_actual[minimum_index]:.3f}",
        xy=(time_s[minimum_index], lambda_actual[minimum_index]),
        xytext=(-46, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6),
        color=INK,
        fontsize=7.8,
    )
    lambda_axis.set_xlabel("time (s)")
    lambda_axis.set_ylabel(r"algebraic connectivity $\lambda_2$")
    lambda_axis.legend(loc="upper right", frameon=False)
    panel_label(lambda_axis, "b")

    barrier_actual = diagnostics["barrier_actual"]
    barrier_nominal = diagnostics["barrier_nominal"]
    for left, right in contiguous_intervals(time_s, barrier_nominal < 0.0):
        barrier_axis.axvspan(left, right, color=PALE_RED, zorder=0)
    barrier_axis.plot(time_s, barrier_actual, color=BLUE, label="applied safe command")
    barrier_axis.plot(time_s, barrier_nominal, color=ORANGE, linestyle=(0, (4, 2)), label="unfiltered nominal command")
    barrier_axis.axhline(0.0, color=INK, linestyle=(0, (2, 2)), linewidth=0.9)
    barrier_axis.set_xlabel("time (s)")
    barrier_axis.set_ylabel(r"exact barrier-matrix margin")
    barrier_axis.legend(loc="upper right", frameon=False)
    panel_label(barrier_axis, "c")

    admitted, rejected, switching = classify_pair_histories(
        distances, config.channel.cutoff_distance
    )
    for order, pair_index in enumerate(admitted):
        distance_axis.plot(
            time_s,
            distances[:, pair_index],
            color=BLUE,
            linewidth=0.85,
            alpha=0.82,
            label="always-admitted pairs" if order == 0 else None,
        )
    for order, pair_index in enumerate(rejected):
        distance_axis.plot(
            time_s,
            distances[:, pair_index],
            color=ORANGE,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label="always-rejected pairs" if order == 0 else None,
        )
    for order, pair_index in enumerate(switching):
        distance_axis.plot(
            time_s,
            distances[:, pair_index],
            color=GOLD,
            linewidth=1.0,
            linestyle=(0, (5, 2, 1, 2)),
            label="range-crossing pairs" if order == 0 else None,
        )
    distance_axis.axhline(config.channel.transition_distance, color=GRAY, linewidth=0.8, linestyle=(0, (5, 2, 1, 2)), label=r"transition $d_t=1.60$ m")
    distance_axis.axhline(config.channel.cutoff_distance, color=INK, linewidth=0.95, linestyle=(0, (2, 2)), label=r"cutoff $d_c=2.12$ m")
    distance_axis.set_xlabel("time (s)")
    distance_axis.set_ylabel("pairwise distance (m)")
    distance_axis.legend(loc="center left", frameon=False)
    panel_label(distance_axis, "d")

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.09, top=0.99, hspace=0.34, wspace=0.28)
    save_figure(figure, output_dir, "physical_experiment_results")


def plot_manuscript_panels(
    run: dict[str, Array | pd.DataFrame],
    diagnostics: dict[str, Array],
    output_dir: Path,
) -> None:
    """Export independent result panels for composition with LaTeX subfloats."""
    config = current_preflight_config()
    time_s = np.asarray(run["time_s"])
    positions = np.asarray(run["positions"])
    references = np.asarray(run["references"])
    distances = diagnostics["pair_distances"]

    trajectory_figure, trajectory_axis = plt.subplots(figsize=(3.45, 2.55))

    for robot in range(4):
        trajectory_axis.plot(
            references[:, robot, 0],
            references[:, robot, 1],
            color=ROBOT_COLORS[robot],
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.62,
        )
        trajectory_axis.plot(
            positions[:, robot, 0],
            positions[:, robot, 1],
            color=ROBOT_COLORS[robot],
            linewidth=1.15,
            label=f"robot {robot + 1}",
        )
        trajectory_axis.plot(
            positions[0, robot, 0],
            positions[0, robot, 1],
            marker="o",
            markersize=3.7,
            markerfacecolor="white",
            markeredgecolor=ROBOT_COLORS[robot],
            markeredgewidth=0.8,
        )
        trajectory_axis.plot(
            positions[-1, robot, 0],
            positions[-1, robot, 1],
            marker="s",
            markersize=3.6,
            color=ROBOT_COLORS[robot],
        )
    for pair_index, (i, j) in enumerate(PAIR_LIST):
        if distances[-1, pair_index] < config.channel.cutoff_distance:
            trajectory_axis.plot(
                [positions[-1, i, 0], positions[-1, j, 0]],
                [positions[-1, i, 1], positions[-1, j, 1]],
                color=MUTED,
                linewidth=0.65,
                alpha=0.65,
                zorder=0,
            )
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.set_xlabel(r"local $x$ (m)")
    trajectory_axis.set_ylabel(r"local $y$ (m)")
    trajectory_axis.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.4,
        fontsize=6.8,
    )
    trajectory_figure.subplots_adjust(left=0.16, right=0.99, bottom=0.18, top=0.99)
    save_figure(
        trajectory_figure, output_dir, "physical_experiment_trajectory"
    )

    connectivity_figure, lambda_axis = plt.subplots(figsize=(3.45, 2.55))
    lambda_actual = diagnostics["lambda_actual"]
    lambda_reference = diagnostics["lambda_reference"]
    for left, right in contiguous_intervals(
        time_s, lambda_reference < config.lambda_req
    ):
        lambda_axis.axvspan(left, right, color=PALE_RED, zorder=0)
    lambda_axis.plot(
        time_s, lambda_actual, color=BLUE, label=r"measured-state $\lambda_2$"
    )
    lambda_axis.plot(
        time_s,
        lambda_reference,
        color=ORANGE,
        linestyle=(0, (4, 2)),
        label=r"reference-state $\lambda_2$",
    )
    lambda_axis.axhline(
        config.lambda_req,
        color=INK,
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        label=r"$\lambda_{\rm req}=0.50$",
    )
    lambda_axis.set_xlabel("time (s)")
    lambda_axis.set_ylabel(r"algebraic connectivity $\lambda_2$")
    lambda_axis.legend(loc="upper right", frameon=False, fontsize=6.8)
    connectivity_figure.subplots_adjust(
        left=0.17, right=0.99, bottom=0.18, top=0.99
    )
    save_figure(
        connectivity_figure, output_dir, "physical_experiment_connectivity"
    )


def summarize(
    run: dict[str, Array | pd.DataFrame],
    diagnostics: dict[str, Array],
) -> dict[str, float | int]:
    config = current_preflight_config()
    frame = run["frame"]
    assert isinstance(frame, pd.DataFrame)
    tracking = diagnostics["tracking"]
    correction = diagnostics["correction"]
    distances = diagnostics["pair_distances"]
    active_counts = np.sum(distances < config.channel.cutoff_distance, axis=1)
    admitted, rejected, switching = classify_pair_histories(
        distances, config.channel.cutoff_distance
    )
    if admitted.size == 0 or rejected.size == 0:
        raise ValueError(
            "reported run must contain at least one always-admitted and one "
            "always-rejected pair"
        )
    mocap_age_columns = [f"car_{index}_mocap_age_s" for index in range(1, 5)]
    node_ritz_columns = [f"node{index}_lambda2_estimate" for index in range(1, 5)]
    node_ritz = frame[node_ritz_columns].to_numpy(float)
    node_ritz_error = np.abs(node_ritz - diagnostics["lambda_actual"][:, None])
    sample_intervals = np.diff(np.asarray(run["time_s"]))
    return {
        "samples": int(len(frame)),
        "minimum_measured_lambda2": float(np.min(diagnostics["lambda_actual"])),
        "minimum_reference_lambda2": float(np.min(diagnostics["lambda_reference"])),
        "measured_below_threshold_samples": int(np.sum(diagnostics["lambda_actual"] < config.lambda_req)),
        "reference_below_threshold_samples": int(np.sum(diagnostics["lambda_reference"] < config.lambda_req)),
        "minimum_exact_applied_barrier_margin": float(np.min(diagnostics["barrier_actual"])),
        "minimum_exact_nominal_barrier_margin": float(np.min(diagnostics["barrier_nominal"])),
        "negative_nominal_barrier_samples": int(np.sum(diagnostics["barrier_nominal"] < 0.0)),
        "minimum_fiedler_gap": float(np.min(diagnostics["fiedler_gap"])),
        "tracking_rmse_m": float(np.sqrt(np.mean(tracking**2))),
        "maximum_tracking_error_m": float(np.max(tracking)),
        "maximum_filter_correction_mps": float(np.max(correction)),
        "maximum_applied_speed_mps": float(np.max(diagnostics["command_speed"])),
        "minimum_pair_distance_m": float(frame["minimum_separation_m"].min()),
        "minimum_geofence_margin_m": float(frame["geofence_margin_m"].min()),
        "maximum_local_solve_time_ms": float(frame["maximum_local_solve_time_ms"].max()),
        "maximum_controller_wall_time_ms": float(frame["controller_wall_time_ms"].max()),
        "maximum_projected_cycle_time_ms": float(frame["virtual_total_time_ms"].max()),
        "deadline_misses": int(frame["deadline_miss"].sum()),
        "mean_control_interval_ms": float(1e3 * np.mean(sample_intervals)),
        "maximum_control_interval_ms": float(1e3 * np.max(sample_intervals)),
        "maximum_mocap_age_ms": float(1e3 * frame[mocap_age_columns].to_numpy(float).max()),
        "maximum_node_ritz_absolute_error": float(np.max(node_ritz_error)),
        "maximum_node_ritz_disagreement": float(frame["node_lambda2_estimate_disagreement"].max()),
        "minimum_active_pair_distance_m": float(np.min(distances[:, admitted])),
        "maximum_active_pair_distance_m": float(np.max(distances[:, admitted])),
        "minimum_filtered_pair_distance_m": float(np.min(distances[:, rejected])),
        "maximum_filtered_pair_distance_m": float(np.max(distances[:, rejected])),
        "range_crossing_pair_count": int(switching.size),
        "minimum_active_edge_count": int(np.min(active_counts)),
        "maximum_active_edge_count": int(np.max(active_counts)),
    }


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace / "四车实验" / "实验结果" / "timeseries.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_figures",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "paper_data"
            / "physical_experiment"
            / "metrics.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_run(args.input)
    diagnostics = reconstruct(run)
    plot_architecture(args.output_dir)
    plot_results(run, diagnostics, args.output_dir)
    plot_manuscript_panels(run, diagnostics, args.output_dir)
    metrics = summarize(run, diagnostics)
    payload = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "source_csv": workspace_relative(args.input),
        "source_csv_sha256": sha256_file(args.input),
        "evaluation_graph": "all six unordered pairs with finite-range cutoff",
        "metrics": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote figures to {args.output_dir.resolve()}")
    print(f"wrote metrics to {args.metrics_output.resolve()}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
