#!/usr/bin/env python3
"""Plot communication scheduling and closed-loop evidence for the current candidate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .current_onboard_profile import communication_round_breakdown


def plot_current_onboard_result(
    data: pd.DataFrame,
    metadata: dict[str, object],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Save the communication-aware four-panel preflight figure."""
    control_period_ms = 1e3 * float(metadata["control_period_s"])
    round_latency_ms = float(metadata["router_round_latency_ms"])
    consensus_rounds = int(metadata["consensus_rounds"])
    breakdown = communication_round_breakdown(
        consensus_rounds,
        record_node_ritz_estimates=bool(metadata["record_node_ritz_estimates"]),
    )
    rounds_per_cycle = sum(breakdown.values())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), constrained_layout=True)

    axis = axes[0, 0]
    cycle_count = int(round(1e3 / control_period_ms))
    colors = plt.get_cmap("tab10").colors
    for cycle in range(cycle_count):
        cycle_start = cycle * control_period_ms
        for round_index in range(rounds_per_cycle):
            start = cycle_start + round_index * round_latency_ms
            axis.broken_barh(
                [(start, 0.92 * round_latency_ms)],
                (0.15, 0.7),
                facecolors=colors[0],
                edgecolors="none",
            )
        axis.axvline(cycle_start, color="0.65", linewidth=0.6, linestyle=":")
    axis.axvline(1e3, color="0.65", linewidth=0.6, linestyle=":")
    axis.set_xlim(0.0, 1e3)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.set_xlabel("Time within one second (ms)")
    axis.set_title(
        f"(a) Communication demand: {rounds_per_cycle * cycle_count} rounds/s",
        loc="left",
    )
    axis.text(
        0.02,
        0.93,
        (
            f"{cycle_count} control cycles/s; {rounds_per_cycle} rounds/cycle; "
            f"{round_latency_ms:g} ms/round"
        ),
        transform=axis.transAxes,
        va="top",
        color="0.3",
        fontsize=7,
    )
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)

    axis = axes[0, 1]
    axis.plot(data["time_s"], data["lambda2"], color="#0072B2", label="$\\lambda_2$")
    axis.plot(data["time_s"], data["lambda3"], color="#D55E00", label="$\\lambda_3$")
    estimate_columns = [f"node{index}_lambda2_estimate" for index in range(1, 5)]
    node_estimates = data[estimate_columns].to_numpy(dtype=float)
    axis.fill_between(
        data["time_s"],
        np.min(node_estimates, axis=1),
        np.max(node_estimates, axis=1),
        color="#56B4E9",
        alpha=0.18,
        linewidth=0.0,
        label="Node Ritz range",
    )
    axis.plot(
        data["time_s"],
        np.mean(node_estimates, axis=1),
        color="#0072B2",
        linewidth=0.9,
        linestyle="-.",
        label="Mean node Ritz",
    )
    axis.axhline(float(metadata["lambda_req"]), color="0.15", linestyle=":", label="$\\lambda_{\\rm req}$")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Laplacian eigenvalue")
    axis.set_title("(b) Connectivity and Ritz diagnostic", loc="left")
    axis.legend(frameon=False, fontsize=6.7, ncol=2)
    axis.grid(alpha=0.18, linewidth=0.5)

    axis = axes[1, 0]
    margin_axis = axis.twinx()
    axis.plot(
        data["time_s"],
        data["subspace_sine_error"],
        color="#009E73",
        linewidth=1.3,
        label="Subspace sine error",
    )
    margin_axis.plot(
        data["time_s"],
        data["exact_command_barrier_margin"],
        color="#CC79A7",
        linewidth=1.2,
        label="Command barrier",
    )
    margin_axis.plot(
        data["time_s"],
        data["exact_realized_velocity_barrier_margin"],
        color="#E69F00",
        linewidth=1.0,
        linestyle="--",
        label="Realized-velocity barrier",
    )
    margin_axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Subspace sine error", color="#009E73")
    margin_axis.set_ylabel("Exact barrier margin")
    axis.tick_params(axis="y", colors="#009E73")
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = margin_axis.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, fontsize=6.7)
    axis.set_title("(c) Estimator and applied-control diagnostics", loc="left")
    axis.grid(alpha=0.18, linewidth=0.5)

    axis = axes[1, 1]
    axis.plot(
        data["time_s"],
        data["controller_wall_time_ms"],
        color="#56B4E9",
        linewidth=1.0,
        label="Measured computation",
    )
    axis.plot(
        data["time_s"],
        data["virtual_total_time_ms"],
        color="#E69F00",
        linewidth=1.2,
        label="Compute + transport",
    )
    axis.plot(
        data["time_s"],
        data["command_application_delay_ms"],
        color="#009E73",
        linewidth=0.9,
        linestyle="--",
        label="Quantized command delay",
    )
    axis.axhline(control_period_ms, color="0.15", linestyle=":", label="Control deadline")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Time (ms)")
    axis.set_title("(d) Cycle timing and command delay", loc="left")
    axis.legend(frameon=False, fontsize=6.7)
    axis.grid(alpha=0.18, linewidth=0.5)

    png_path = output_directory / "fig_current_onboard_candidate.png"
    pdf_path = output_directory / "fig_current_onboard_candidate.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.output_directory / "preflight_timeseries.csv")
    metadata = json.loads(
        (args.output_directory / "preflight_config.json").read_text(encoding="utf-8")
    )
    plot_current_onboard_result(data, metadata, args.output_directory)


if __name__ == "__main__":
    main()
