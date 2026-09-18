#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the publication-polished figures used by the current manuscript.

The visual system targets IEEE journal production: embedded TrueType
fonts, 8--10 pt typography at final size, restrained color, grayscale-safe
line encodings, formal scientific notation, and uncluttered inset layouts.

The crossing and frozen-recovery panels read the paper's core records.
Fig. 3 reads the topology-disconnection stress-test record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from run_paper_experiments import TOPOLOGY_GOAL, TOPOLOGY_INITIAL, channel_graph, rectangle_positions
from violation_free_soc_core import MoxChannelParameters, complete_topology, exact_low_basis


Array = np.ndarray


def positions_from_row(row: pd.Series, number_of_robots: int = 6) -> Array:
    return np.asarray(
        [[row[f"p{i}_x_m"], row[f"p{i}_y_m"]] for i in range(1, number_of_robots + 1)],
        dtype=float,
    )


def align_basis_to_reference(basis: Array, reference: Array) -> tuple[Array, Array]:
    left, _, right_t = np.linalg.svd(basis.T @ reference)
    rotation = left @ right_t
    return basis @ rotation, rotation


# Colorblind-safe, restrained IEEE palette.
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
PALE_GRAY = "#F2F4F5"


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
        "legend.fontsize": 8.0,
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
        "lines.linewidth": 1.00,
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


METHOD_STYLES = {
    "nominal": dict(color=GRAY, linestyle=(0, (4, 2)), linewidth=0.85, label="nominal"),
    "centralized": dict(color=BLUE, linestyle=(0, (5, 1.6, 1.2, 1.6)), linewidth=0.95, label="centralized"),
    "allocation": dict(color=ORANGE, linestyle="-", linewidth=1.25, label="proposed"),
}


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
    plt.close(figure)


def save_panel_groups(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    groups: tuple[tuple[str, tuple[plt.Axes, ...]], ...],
) -> None:
    """Export caption-free panel crops for assembly with LaTeX ``subfig``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for suffix, axes in groups:
        display_bbox = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
        inches_bbox = display_bbox.transformed(figure.dpi_scale_trans.inverted()).padded(0.025)
        for extension, dpi in (("pdf", None), ("png", 300)):
            figure.savefig(
                output_dir / f"{stem}_{suffix}.{extension}",
                dpi=dpi,
                bbox_inches=inches_bbox,
                pad_inches=0.0,
                facecolor="white",
            )


def panel_caption(
    axis: plt.Axes,
    label: str,
    title: str,
    *,
    y: float = -0.28,
) -> None:
    """Place one uniform IEEE-style panel caption below an axis."""
    axis.text(
        0.5,
        y,
        rf"$\bf{{({label})}}$ {title}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK,
        clip_on=False,
    )


def group_caption(
    figure: plt.Figure,
    axes: tuple[plt.Axes, ...],
    label: str,
    title: str,
    *,
    y_offset: float = 0.055,
) -> None:
    """Center one panel caption below a group of peer axes."""
    boxes = [axis.get_position() for axis in axes]
    x_center = 0.5 * (min(box.x0 for box in boxes) + max(box.x1 for box in boxes))
    y_bottom = min(box.y0 for box in boxes)
    figure.text(
        x_center,
        y_bottom - y_offset,
        rf"$\bf{{({label})}}$ {title}",
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK,
    )


def clean_diagram_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def sci_tex(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def add_badge(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = INK,
    facecolor: str = "white",
    transform=None,
    horizontalalignment: str = "center",
    verticalalignment: str = "center",
    fontsize: float = 8.0,
) -> None:
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes if transform is None else transform,
        ha=horizontalalignment,
        va=verticalalignment,
        fontsize=fontsize,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.23",
            facecolor=facecolor,
            edgecolor="none",
            alpha=0.96,
        ),
        zorder=8,
    )


def draw_subspace_geometry(axis: plt.Axes) -> None:
    square = np.asarray([[0.62, 0.62], [-0.62, 0.62], [-0.62, -0.62], [0.62, -0.62]])
    angle = np.deg2rad(31.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = square @ rotation
    closed = np.r_[np.arange(4), 0]

    axis.fill(square[closed, 0], square[closed, 1], color=PALE_BLUE, alpha=0.95, zorder=0)
    axis.plot(square[closed, 0], square[closed, 1], color=BLUE, linewidth=1.05)
    axis.plot(rotated[closed, 0], rotated[closed, 1], color=GREEN, linestyle=(0, (4, 2)), linewidth=0.95)
    axis.scatter(square[:, 0], square[:, 1], s=31, color="white", edgecolor=BLUE, linewidth=0.75, zorder=4)
    axis.scatter(rotated[:, 0], rotated[:, 1], s=28, color="white", edgecolor=GREEN, linewidth=0.70, zorder=4)
    for index, point in enumerate(square, start=1):
        axis.text(point[0] + 0.08, point[1] + 0.07, str(index), fontsize=7.8, color=INK)

    axis.axhline(0.0, color=GRID, linewidth=0.50)
    axis.axvline(0.0, color=GRID, linewidth=0.50)
    axis.annotate(
        "",
        xy=(1.08, 0.0),
        xytext=(-1.08, 0.0),
        arrowprops=dict(arrowstyle="<->", color=SKY, linewidth=1.00, linestyle="--"),
    )
    axis.annotate(
        "",
        xy=(0.0, 1.08),
        xytext=(0.0, -1.08),
        arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.00, linestyle="-."),
    )
    axis.text(0.60, 0.08, r"selected $v_2^{-}$", color=BLUE, fontsize=8.0)
    axis.text(0.08, 0.70, r"selected $v_2^{+}$", color=ORANGE, fontsize=8.0, rotation=90, va="center")
    add_badge(
        axis,
        0.96,
        0.91,
        r"$\mathrm{span}(VQ)=\mathrm{span}(V)$",
        color="#176149",
        facecolor=PALE_GREEN,
        horizontalalignment="right",
        fontsize=7.5,
    )
    axis.text(
        0.5,
        0.025,
        "one direction may switch labels\n2-D invariant subspace persists",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(-1.28, 1.28)
    axis.set_aspect("equal", adjustable="box")
    clean_diagram_axis(axis)


def draw_lorentz_section(axis: plt.Axes) -> None:
    rho = np.linspace(0.0, 1.30, 200)
    upper = np.full_like(rho, 1.34)
    axis.fill_between(rho, rho, upper, color=PALE_BLUE, alpha=0.95, zorder=0)
    axis.plot(rho, rho, color=BLUE, linewidth=1.05)
    axis.plot([0.0, 0.0], [0.0, 1.34], color=INK, linewidth=0.60)

    points = [
        (0.38, 0.95, GREEN, r"$M\succ0$"),
        (0.78, 0.78, GOLD, r"$\lambda_{\min}(M)=0$"),
        (0.96, 0.56, "#B23A48", "outside cone"),
    ]
    offsets = [(0.06, 0.04), (0.04, 0.08), (-0.27, -0.11)]
    for (x_value, y_value, color, label), (dx, dy) in zip(points, offsets):
        axis.scatter([x_value], [y_value], s=34, color=color, edgecolor="white", linewidth=0.55, zorder=4)
        axis.text(x_value + dx, y_value + dy, label, color=color, fontsize=7.8)

    add_badge(axis, 0.53, 0.90, r"$M\succeq0\ \Longleftrightarrow\ s\geq\rho$", facecolor="white", color=INK)
    axis.text(
        0.5,
        0.06,
        r"congruence $Q^\top M Q$ preserves $(s,\rho)$",
        transform=axis.transAxes,
        ha="center",
        fontsize=7.4,
        color=MUTED,
    )
    axis.set_xlabel(r"transverse radius $\rho=\|z\|_2$")
    axis.set_ylabel(r"trace coordinate $s$")
    axis.set_xlim(-0.02, 1.34)
    axis.set_ylim(-0.02, 1.36)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.grid(alpha=0.34)


def cone_icon(axis: plt.Axes, center: tuple[float, float], scale: float, color: str) -> None:
    x_value, y_value = center
    patch = Polygon(
        [[x_value, y_value], [x_value - scale, y_value + 1.45 * scale], [x_value + scale, y_value + 1.45 * scale]],
        closed=True,
        facecolor=PALE_GREEN,
        edgecolor=GREEN,
        linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.annotate(
        "",
        xy=(x_value + 0.10 * scale, y_value + 1.02 * scale),
        xytext=(x_value, y_value + 0.08 * scale),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.05),
    )


def draw_zero_sum_geometry(axis: plt.Axes) -> None:
    add_badge(axis, 0.50, 0.89, r"$\sum_i r_i(q)=0$", color="#176149", facecolor=PALE_GREEN)

    centers = [1.1, 2.75, 4.40]
    colors = [BLUE, ORANGE, GREEN]
    for index, (x_value, color) in enumerate(zip(centers, colors), start=1):
        cone_icon(axis, (x_value, 2.25), 0.43, color)
        axis.text(x_value, 1.93, rf"$\psi_{index}-r_{index}$", ha="center", fontsize=8.0)
        if index < 3:
            axis.text(x_value + 0.82, 2.80, "+", ha="center", fontsize=11, color=MUTED)
    axis.annotate(
        "",
        xy=(6.05, 2.80),
        xytext=(5.02, 2.80),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0),
    )
    cone_icon(axis, (7.00, 2.02), 0.70, ORANGE)
    axis.text(7.00, 1.68, r"$\sum_i\psi_i\in\mathcal{Q}_3$", ha="center", fontsize=8.3)

    axis.plot([1.05, 7.28], [0.68, 0.68], color="#AAB2B9", linewidth=1.0)
    axis.scatter([1.42], [0.68], s=34, color=ORANGE, edgecolor="white", linewidth=0.55, zorder=4)
    axis.scatter([6.92], [0.68], s=48, color=BLUE, marker="*", zorder=4)
    axis.annotate(
        "",
        xy=(6.65, 0.68),
        xytext=(1.72, 0.68),
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.0),
    )
    axis.text(1.42, 0.38, r"$u^{(m)}$: constraints met", ha="center", fontsize=7.5, color=ORANGE)
    axis.text(6.92, 0.38, r"$u^\star_{\rm cen}$: optimal", ha="center", fontsize=8.0, color=BLUE)
    axis.text(
        4.15,
        0.04,
        "global barrier is satisfied before fixed-state convergence",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=MUTED,
    )
    axis.set_xlim(0.25, 7.85)
    axis.set_ylim(-0.02, 3.95)
    clean_diagram_axis(axis)


def plot_geometric_overview(output_dir: Path) -> None:
    figure = plt.figure(figsize=(7.16, 3.08))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.04, 1.02, 1.30), wspace=0.27)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
    )
    draw_subspace_geometry(axes[0])
    draw_lorentz_section(axes[1])
    draw_zero_sum_geometry(axes[2])
    figure.subplots_adjust(left=0.055, right=0.995, bottom=0.23, top=0.98)
    save_panel_groups(
        figure,
        output_dir,
        "geometric_overview",
        tuple((label, (axis,)) for label, axis in zip(("a", "b", "c"), axes)),
    )
    panel_caption(axes[0], "a", "Basis-covariant spectral cluster")
    panel_caption(axes[1], "b", "Exact PSD--Lorentz geometry")
    # This diagram uses the full subplot height, whereas panels (a)--(b) use
    # equal-aspect boxes.  A shallower offset aligns the legacy composite.
    panel_caption(axes[2], "c", "Zero-sum allocation", y=-0.11)
    save_figure(figure, output_dir, "geometric_overview")


def crossing_lines(axis: plt.Axes, crossings: Array) -> None:
    for crossing in crossings:
        axis.axvline(crossing, color=GOLD, linewidth=0.55, alpha=0.62, zorder=0)


def draw_spectral_snapshot(
    axis: plt.Axes,
    step: int,
    reference_basis: Array,
    channel: MoxChannelParameters,
    *,
    highlighted: bool,
) -> None:
    positions, _, _ = rectangle_positions(step)
    basis, values, _ = exact_low_basis(
        positions, complete_topology(4), sigma=1.0, channel_parameters=channel
    )
    aligned, rotation = align_basis_to_reference(basis, reference_basis)
    coordinate = rotation[0, :]
    coordinate /= max(float(np.linalg.norm(coordinate)), 1e-12)
    closed = np.r_[np.arange(4), 0]

    axis.fill(aligned[closed, 0], aligned[closed, 1], color=PALE_BLUE, alpha=0.95)
    axis.plot(aligned[closed, 0], aligned[closed, 1], color=BLUE, linewidth=1.00)
    axis.scatter(aligned[:, 0], aligned[:, 1], s=31, color="white", edgecolor=BLUE, linewidth=0.70, zorder=4)
    for index, point in enumerate(aligned, start=1):
        axis.text(point[0] + 0.035, point[1] + 0.038, str(index), fontsize=7.7, color=INK)
    extent = 0.74
    axis.annotate(
        "",
        xy=extent * coordinate,
        xytext=-extent * coordinate,
        arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.05, linestyle="--"),
    )
    axis.axhline(0.0, color=GRID, linewidth=0.50)
    axis.axvline(0.0, color=GRID, linewidth=0.50)
    if highlighted:
        add_badge(axis, 0.96, 0.92, r"$v_2$ is not unique", color="#8A5200", facecolor=PALE_GOLD, horizontalalignment="right", fontsize=7.5)
    else:
        axis.text(0.96, 0.90, r"ordered $v_2$ axis", transform=axis.transAxes, ha="right", fontsize=7.5, color=ORANGE)
    axis.text(
        0.04,
        0.06,
        rf"$\lambda_3-\lambda_2={sci_tex(abs(values[2]-values[1]), 1)}$",
        transform=axis.transAxes,
        fontsize=7.4,
        color=MUTED,
    )
    axis.set_xlim(-0.80, 0.80)
    axis.set_ylim(-0.80, 0.80)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    if highlighted:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(GOLD)
            spine.set_linewidth(0.72)


def plot_crossing_evidence(
    data: pd.DataFrame,
    output_dir: Path,
    channel: MoxChannelParameters,
) -> None:
    selected = data[data["seed"] == 1]
    displayed = selected[selected["step"] >= 60]
    block = displayed[displayed["method"] == "ours-r2-block"].sort_values("step")
    crossings = block.loc[np.abs(block["fiedler_gap"]) < 1e-10, "step"].to_numpy()

    figure = plt.figure(figsize=(7.16, 5.18))
    grid = figure.add_gridspec(2, 6, height_ratios=(0.92, 1.08), hspace=0.86, wspace=0.48)
    axis_spectrum = figure.add_subplot(grid[0, 0:3])
    axis_error = figure.add_subplot(grid[0, 3:6])
    spectral_axes = [figure.add_subplot(grid[1, start : start + 2]) for start in (0, 2, 4)]

    step = block["step"].to_numpy()
    lambda2 = block["lambda2_true"].to_numpy()
    lambda3 = block["lambda3_true"].to_numpy()
    axis_spectrum.fill_between(step, lambda2, lambda3, color=PALE_BLUE, alpha=0.95, zorder=0)
    axis_spectrum.plot(step, lambda2, color=INK, label=r"true $\lambda_2$", zorder=3)
    axis_spectrum.plot(step, lambda3, color=MUTED, linestyle=(0, (4, 2)), label=r"true $\lambda_3$", zorder=2)
    axis_spectrum.plot(
        step,
        block["lambda2_estimate_mean"],
        color=ORANGE,
        linewidth=1.25,
        marker="o",
        markevery=15,
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.65,
        label=r"proposed $r=2$",
        zorder=4,
    )
    crossing_row = block.loc[block["step"] == 180].iloc[0]
    axis_spectrum.annotate(
        r"zero eigengap: $2.8\times10^{-17}$",
        xy=(180, float(crossing_row["lambda2_true"])),
        xytext=(197, 0.273),
        arrowprops=dict(arrowstyle="->", color=GOLD, linewidth=0.85),
        fontsize=7.7,
        color="#755300",
        bbox=dict(boxstyle="round,pad=0.20", facecolor=PALE_GOLD, edgecolor="none"),
    )
    crossing_lines(axis_spectrum, crossings)
    axis_spectrum.set_ylabel("eigenvalue")
    axis_spectrum.set_xlabel("sampling instant")
    axis_spectrum.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.3, columnspacing=1.2)

    estimator_styles = {
        "ours-r2-block": (ORANGE, "-", 1.25, r"proposed $r=2$"),
        "ours-r1-ablation": (SKY, "-.", 0.95, r"$r=1$ ablation"),
        "yang-2010-single-fiedler": (GREEN, (0, (4, 2)), 0.90, "Yang et al."),
    }
    axis_error.axhspan(-0.002, 0.002, color=PALE_GREEN, alpha=0.95, zorder=0)
    for method, (color, linestyle, linewidth, label) in estimator_styles.items():
        subset = displayed[displayed["method"] == method].sort_values("step")
        axis_error.plot(
            subset["step"],
            subset["lambda2_estimate_mean"] - subset["lambda2_true"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=4 if method == "ours-r2-block" else 2,
        )
    crossing_lines(axis_error, crossings)
    axis_error.axhline(0.0, color=INK, linewidth=0.62, alpha=0.72)
    add_badge(
        axis_error,
        0.97,
        0.07,
        r"$r=2$ RMSE $=5.30\times10^{-4}$",
        color="#176149",
        facecolor=PALE_GREEN,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=7.7,
    )
    axis_error.set_ylabel("estimate minus truth")
    axis_error.set_xlabel("sampling instant")
    axis_error.legend(loc="upper right", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.1)
    for axis in (axis_spectrum, axis_error):
        axis.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    snapshot_steps = [165, 180, 195]
    reference_positions, _, _ = rectangle_positions(snapshot_steps[0])
    reference_basis, _, _ = exact_low_basis(
        reference_positions, complete_topology(4), sigma=1.0, channel_parameters=channel
    )
    labels = ["c", "d", "e"]
    titles = ["Before crossing", "At multiplicity two", "After crossing"]
    for index, (axis, sample) in enumerate(zip(spectral_axes, snapshot_steps)):
        draw_spectral_snapshot(
            axis,
            sample,
            reference_basis,
            channel,
            highlighted=(index == 1),
        )
        axis.set_xlabel("spectral coordinate 1")
    spectral_axes[0].set_ylabel("spectral coordinate 2")
    spectral_axes[1].set_yticklabels([])
    spectral_axes[2].set_yticklabels([])
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.985)
    panel_axes = (axis_spectrum, axis_error, *spectral_axes)
    save_panel_groups(
        figure,
        output_dir,
        "crossing_evidence",
        tuple((label, (axis,)) for label, axis in zip(("a", "b", "c", "d", "e"), panel_axes)),
    )
    panel_caption(axis_spectrum, "a", "Ordered spectral branches", y=-0.30)
    panel_caption(axis_error, "b", "Signed estimation error", y=-0.30)
    for axis, label, title in zip(spectral_axes, labels, titles):
        panel_caption(axis, label, title, y=-0.29)
    save_figure(figure, output_dir, "crossing_evidence")


def shade_negative(axis: plt.Axes, values: Array) -> None:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = max(maximum - minimum, 1e-6)
    lower = min(minimum - 0.05 * span, -0.02 * span)
    upper = maximum + 0.07 * span
    axis.set_ylim(lower, upper)
    axis.axhspan(lower, 0.0, color=PALE_RED, alpha=0.92, zorder=0)
    axis.axhline(0.0, color="#9A3A45", linewidth=0.72, zorder=1)


def draw_terminal_pair(
    axes: tuple[plt.Axes, plt.Axes],
    proposed: pd.DataFrame,
    nominal: pd.DataFrame,
    channel: MoxChannelParameters,
) -> None:
    """Show faint physical paths and weighted terminal graphs at one scale."""
    trajectories: dict[str, Array] = {}
    terminal_rows: dict[str, pd.Series] = {}
    for name, subset in (("proposed", proposed), ("nominal", nominal)):
        ordered = subset.sort_values("time_s")
        sampled = np.asarray(
            [positions_from_row(row) for _, row in ordered.iterrows()],
            dtype=float,
        )
        trajectories[name] = np.concatenate((TOPOLOGY_INITIAL[None, :, :], sampled), axis=0)
        terminal_rows[name] = ordered.iloc[-1]

    all_positions = np.concatenate((*trajectories.values(), TOPOLOGY_GOAL[None, :, :]), axis=0)
    x_min, x_max = np.min(all_positions[:, :, 0]), np.max(all_positions[:, :, 0])
    y_min, y_max = np.min(all_positions[:, :, 1]), np.max(all_positions[:, :, 1])
    x_pad = 0.045 * (x_max - x_min)
    y_pad = 0.075 * (y_max - y_min)
    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    layouts: dict[str, list[tuple[int, int, float]]] = {}
    maximum_weight = 1e-12
    for name, row in terminal_rows.items():
        positions = positions_from_row(row)
        edges, _, _, _ = channel_graph(positions, channel)
        positive = [(i, j, weight) for i, j, weight in edges if weight > 0.0]
        layouts[name] = positive
        if positive:
            maximum_weight = max(maximum_weight, max(weight for _, _, weight in positive))

    specifications = (
        (axes[0], "proposed", ORANGE),
        (axes[1], "nominal", GRAY),
    )
    for axis, name, color in specifications:
        path = trajectories[name]
        terminal = path[-1]
        row = terminal_rows[name]
        for robot in range(path.shape[1]):
            axis.plot(
                path[:, robot, 0],
                path[:, robot, 1],
                color=color,
                linewidth=0.62,
                linestyle=(0, (2.0, 1.6)),
                alpha=0.18,
                zorder=1,
            )
        axis.scatter(
            path[0, :, 0],
            path[0, :, 1],
            s=20,
            facecolor="white",
            edgecolor="#8E979E",
            linewidth=0.72,
            alpha=0.90,
            zorder=2,
        )
        axis.scatter(
            TOPOLOGY_GOAL[:, 0],
            TOPOLOGY_GOAL[:, 1],
            s=38,
            marker="D",
            facecolor="none",
            edgecolor="#59656E",
            linewidth=0.62,
            alpha=0.58,
            zorder=2.5,
        )
        axis.scatter(
            [TOPOLOGY_GOAL[0, 0]],
            [TOPOLOGY_GOAL[0, 1]],
            s=50,
            marker="D",
            facecolor="none",
            edgecolor=INK,
            linewidth=0.82,
            alpha=0.88,
            zorder=2.6,
        )
        for i, j, weight in layouts[name]:
            normalized = weight / maximum_weight
            axis.plot(
                terminal[[i, j], 0],
                terminal[[i, j], 1],
                color=color,
                linewidth=0.45 + 1.65 * normalized,
                alpha=0.52 + 0.48 * normalized,
                zorder=3,
            )
        axis.scatter(
            terminal[:, 0],
            terminal[:, 1],
            s=31,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        for robot, (x_m, y_m) in enumerate(terminal, start=1):
            axis.text(
                x_m,
                y_m,
                str(robot),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                fontweight="semibold",
                zorder=5,
            )
        axis.text(
            0.5,
            1.02,
            name.capitalize(),
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
            fontweight="semibold",
            clip_on=False,
        )
        if name == "proposed":
            margin = float(row["connectivity_safety_margin"])
            axis.text(
                0.03,
                0.97,
                rf"$h_{{\rm conn}}={margin:+.3f}$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                color="#176149",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
                zorder=7,
            )
        axis.annotate(
            r"$g_1$",
            xy=TOPOLOGY_GOAL[0],
            xytext=(4.0, 2.2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6.0,
            color=INK,
            zorder=6,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        clean_diagram_axis(axis)

def plot_all_iterate_safety(
    data: pd.DataFrame,
    output_dir: Path,
    channel: MoxChannelParameters,
) -> None:
    compared = data[data["method"].isin(METHOD_STYLES)].copy()
    proposed = compared[compared["method"] == "allocation"].sort_values("time_s")
    nominal = compared[compared["method"] == "nominal"].sort_values("time_s")
    event_mask = (proposed["edge_appearances"] > 0) | (proposed["edge_disappearances"] > 0)
    event_rows = proposed[event_mask]
    event_start = float(event_rows["time_s"].min())
    event_end = float(event_rows["time_s"].max())
    event_count = int(event_rows["edge_appearances"].sum() + event_rows["edge_disappearances"].sum())

    figure = plt.figure(figsize=(7.16, 4.02))
    grid = figure.add_gridspec(2, 12, height_ratios=(1.15, 1.0), hspace=0.84, wspace=0.70)
    axis_safety = figure.add_subplot(grid[0, :])
    axis_matrix = figure.add_subplot(grid[1, 0:5])
    terminal_grid = grid[1, 5:12].subgridspec(1, 2, wspace=0.06)
    terminal_axes = (
        figure.add_subplot(terminal_grid[0, 0]),
        figure.add_subplot(terminal_grid[0, 1]),
    )

    for method, style in METHOD_STYLES.items():
        subset = compared[compared["method"] == method].sort_values("time_s")
        axis_safety.plot(subset["time_s"], subset["connectivity_safety_margin"], **style)
    shade_negative(axis_safety, compared["connectivity_safety_margin"].to_numpy())
    axis_safety.axvspan(event_start, event_end, color=PALE_GRAY, alpha=0.48, zorder=0)
    axis_safety.axvline(event_start, color=MUTED, linestyle=":", linewidth=0.52)
    axis_safety.axvline(event_end, color=MUTED, linestyle=":", linewidth=0.52)
    axis_safety.text(
        0.31,
        0.78,
        f"{event_count} support changes",
        transform=axis_safety.transAxes,
        ha="center",
        va="top",
        fontsize=7.3,
        color=MUTED,
    )
    negative_nominal = int((nominal["connectivity_safety_margin"] < 0).sum())
    negative_proposed = int((proposed["connectivity_safety_margin"] < 0).sum())
    axis_safety.set_ylabel(r"$h_{\rm conn}=\lambda_2-\lambda_{\rm req}$")
    axis_safety.set_xlabel("time (s)")
    axis_safety.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.35,
        columnspacing=1.25,
        borderaxespad=0.0,
    )

    axis_zoom = axis_safety.inset_axes([0.56, 0.16, 0.42, 0.38])
    for method, style in METHOD_STYLES.items():
        subset = compared[compared["method"] == method].sort_values("time_s")
        zoom_style = dict(style)
        zoom_style.pop("label", None)
        zoom_style["linewidth"] = max(0.70, float(zoom_style["linewidth"]) * 0.82)
        axis_zoom.plot(subset["time_s"], subset["connectivity_safety_margin"], **zoom_style)
    axis_zoom.set_xlim(max(0.45, float(compared["time_s"].min())), float(compared["time_s"].max()))
    zoom_lower = min(-0.205, 1.04 * float(nominal["connectivity_safety_margin"].min()))
    axis_zoom.set_ylim(zoom_lower, 0.035)
    axis_zoom.axhspan(zoom_lower, 0.0, color=PALE_RED, alpha=0.84, zorder=0)
    axis_zoom.axhline(0.0, color="#9A3A45", linewidth=0.58, zorder=1)
    axis_zoom.set_title("late-time status", loc="left", pad=2, fontsize=7.0, color=MUTED)
    axis_zoom.tick_params(axis="both", labelsize=6.8, length=2.2, width=0.45)
    axis_zoom.grid(True, linewidth=0.25, alpha=0.40)
    axis_zoom.text(
        0.98,
        0.77,
        rf"proposed: {negative_proposed}/{len(proposed)}; min $={proposed['connectivity_safety_margin'].min():.4f}$",
        transform=axis_zoom.transAxes,
        ha="right",
        va="center",
        fontsize=6.8,
        color=ORANGE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.8),
    )
    axis_zoom.text(
        0.98,
        0.13,
        rf"nominal: {negative_nominal}/{len(nominal)}; terminal $\lambda_2=0$",
        transform=axis_zoom.transAxes,
        ha="right",
        va="center",
        fontsize=6.8,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.8),
    )

    time = proposed["time_s"].to_numpy()
    matrix = proposed["exact_global_matrix_margin"].to_numpy()
    if np.any(matrix <= 0.0):
        raise ValueError("The proposed exact assembled-cone margin must be positive for the log-scale panel.")
    minimum_index = int(np.nanargmin(matrix))
    minimum_matrix = float(matrix[minimum_index])
    axis_matrix.plot(time, matrix, color=ORANGE, linewidth=1.12)
    axis_matrix.set_yscale("log")
    axis_matrix.set_ylim(0.68 * minimum_matrix, 1.45 * float(np.nanmax(matrix)))
    axis_matrix.axhline(minimum_matrix, color=GREEN, linestyle=":", linewidth=0.72, zorder=1)
    axis_matrix.scatter(
        [time[minimum_index]],
        [minimum_matrix],
        s=17,
        color="white",
        edgecolor=GREEN,
        linewidth=0.70,
        zorder=4,
    )
    axis_matrix.grid(True, which="major", linewidth=0.30, alpha=0.46)
    axis_matrix.grid(False, which="minor")
    add_badge(
        axis_matrix,
        0.97,
        0.90,
        rf"min $={sci_tex(minimum_matrix)}>0$",
        color="#176149",
        facecolor=PALE_GREEN,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=7.2,
    )
    axis_matrix.set_ylabel(r"$\lambda_{\min}(\mathcal{M})$")
    axis_matrix.set_xlabel("time (s)")

    draw_terminal_pair(terminal_axes, proposed, nominal, channel)

    figure.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.985)
    save_panel_groups(
        figure,
        output_dir,
        "all_iterate_safety",
        (
            ("a", (axis_safety,)),
            ("b", (axis_matrix,)),
            ("c", terminal_axes),
        ),
    )
    panel_caption(axis_safety, "a", "Sampled-time connectivity margin", y=-0.31)
    panel_caption(axis_matrix, "b", "Exact cone margin", y=-0.34)
    group_caption(
        figure,
        terminal_axes,
        "c",
        "Terminal configurations",
        y_offset=0.087,
    )
    save_figure(figure, output_dir, "all_iterate_safety")


def draw_double_root_inset(axis: plt.Axes) -> None:
    angles = np.arange(6) * np.pi / 3.0
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    for index in range(6):
        neighbor = (index + 1) % 6
        axis.plot(positions[[index, neighbor], 0], positions[[index, neighbor], 1], color=BLUE, linewidth=0.85)
    axis.scatter(positions[:, 0], positions[:, 1], s=16, color=ORANGE, edgecolor="white", linewidth=0.4, zorder=3)
    axis.text(
        0.5,
        0.02,
        r"$\lambda_2=\lambda_3=0.2142$" + "\n" + r"gap $=1.94\times10^{-16}$",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#755300",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#D7C78E")
        spine.set_linewidth(0.65)


def plot_repeated_root_recovery(data: pd.DataFrame, output_dir: Path) -> None:
    figure, (axis_control, axis_residual) = plt.subplots(1, 2, figsize=(7.16, 2.82))
    iteration = data["iteration"].to_numpy()
    control_error = np.maximum(data["control_error_to_central"].to_numpy(), 1e-17)
    residual = np.maximum(data["master_residual_eta_q"].to_numpy(), 1e-17)
    objective_gap = np.abs(data["objective_gap"].to_numpy())
    matrix_margin = data["exact_global_matrix_margin"].to_numpy()

    axis_control.semilogy(iteration, control_error, color=ORANGE, linewidth=1.20)
    axis_control.scatter(
        [iteration[0], iteration[-1]],
        [control_error[0], control_error[-1]],
        s=22,
        color=[GRAY, ORANGE],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    axis_control.annotate(
        r"start $=99.3$",
        xy=(iteration[0], control_error[0]),
        xytext=(720, 2.1e2),
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
        fontsize=7.8,
    )
    axis_control.annotate(
        r"final $=4.15\times10^{-9}$",
        xy=(iteration[-1], control_error[-1]),
        xytext=(3150, 1.6e-6),
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.8),
        fontsize=7.8,
        color="#944000",
    )
    inset = axis_control.inset_axes([0.63, 0.56, 0.34, 0.37])
    draw_double_root_inset(inset)
    panel_caption(axis_control, "a", "Centralized control recovery", y=-0.29)
    axis_control.set_ylabel(r"$\|u^m-u^\star_{\rm cen}\|_2$")
    axis_control.set_xlabel("allocation iteration")

    axis_residual.semilogy(iteration, residual, color=ORANGE, linewidth=1.20)
    axis_residual.axhspan(1e-17, 1e-12, color=PALE_GRAY, alpha=0.92, zorder=0)
    axis_residual.scatter(
        [iteration[0], iteration[-1]],
        [residual[0], residual[-1]],
        s=22,
        color=[GRAY, ORANGE],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    axis_residual.annotate(
        rf"final $\eta_q={sci_tex(float(residual[-1]))}$",
        xy=(iteration[-1], residual[-1]),
        xytext=(3300, 2.0e-5),
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.8),
        fontsize=7.8,
        color="#944000",
    )
    add_badge(
        axis_residual,
        0.97,
        0.93,
        rf"final $|f-f^\star|={sci_tex(float(objective_gap[-1]))}$"
        + "\n"
        + rf"min $\lambda_{{\min}}(\mathcal{{M}})={sci_tex(float(matrix_margin.min()))}$",
        color="#176149",
        facecolor="white",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=7.5,
    )
    panel_caption(axis_residual, "b", "Fixed-state subgradient norm", y=-0.29)
    axis_residual.set_ylabel(r"subgradient norm $\eta_q$")
    axis_residual.set_xlabel("allocation iteration")
    for axis in (axis_control, axis_residual):
        axis.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.98, wspace=0.29)
    save_figure(figure, output_dir, "repeated_root_recovery")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=root / "paper_data" / "core")
    parser.add_argument("--topology-dir", type=Path, default=root / "paper_data" / "topology")
    parser.add_argument("--output-dir", type=Path, default=root / "paper_figures")
    args = parser.parse_args()

    crossing = pd.read_csv(args.input_dir / "rectangle_crossing.csv")
    topology = pd.read_csv(args.topology_dir / "topology_formation.csv")
    repeated = pd.read_csv(args.input_dir / "repeated_root_recovery.csv")
    channel = MoxChannelParameters()

    plot_geometric_overview(args.output_dir)
    plot_crossing_evidence(crossing, args.output_dir, channel)
    plot_all_iterate_safety(topology, args.output_dir, channel)
    plot_repeated_root_recovery(repeated, args.output_dir)
    print(f"Wrote line-weight-refined manuscript figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
