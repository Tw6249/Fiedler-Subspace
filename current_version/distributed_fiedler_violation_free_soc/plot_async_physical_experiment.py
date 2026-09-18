#!/usr/bin/env python3
"""Generate figures and traceable metrics for the asynchronous physical run.

The retained telemetry is long-form: each row is one robot's local report.
Trajectory-level statistics use only recorded rows.  The connectivity panel
aligns reports by the intended local cycle and linearly interpolates missing
per-agent reports for visualization; the manuscript's reported minimum
connectivity comes from the supervisor summary rather than this interpolation.

The publication architecture panel is manually authored.  Routine telemetry
regeneration preserves it; use ``--regenerate-architecture`` only when the
fallback Matplotlib architecture diagram is intentionally requested.
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
from violation_free_soc_core import exact_low_basis


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    WORKSPACE
    / "最新实物实验"
    / "动捕数据记录"
    / "run_20260821_194553_719606(实机实验)"
)
PAIR_LIST = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
ROBOT_NAMES = ("car_1", "car_2", "car_3", "car_4")
ROBOT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
INK = "#20252A"
MUTED = "#65717C"
GRID = "#D9DEE2"
PALE_BLUE = "#E8F2F7"
PALE_GREEN = "#E7F3ED"
PALE_GOLD = "#FCF2D8"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.4,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.32,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(figure)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    paths = {
        name: run_dir / name
        for name in ("timeseries.csv", "config.json", "summary.json", "events.csv")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(paths["timeseries.csv"])
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    events = pd.read_csv(paths["events.csv"])
    expected = {
        "agent",
        "local_cycle",
        "local_elapsed_s",
        "local_x_m",
        "local_y_m",
        "reference_x_m",
        "reference_y_m",
        "cycle_wall_time_ms",
        "neighbor_ids",
        "local_soc_margin",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"missing telemetry columns: {sorted(missing)}")
    return frame, config, summary, events


def cycle_aligned(frame: pd.DataFrame, samples: int = 200) -> dict[str, np.ndarray]:
    cycles = np.arange(samples)
    positions = np.empty((samples, 4, 2))
    references = np.empty((samples, 4, 2))
    observed = np.zeros((samples, 4), dtype=bool)
    for robot, name in enumerate(ROBOT_NAMES):
        local = (
            frame.loc[frame["agent"] == name]
            .sort_values("local_cycle")
            .drop_duplicates("local_cycle", keep="last")
            .set_index("local_cycle")
        )
        observed[local.index.to_numpy(int), robot] = True
        numeric = local[
            ["local_x_m", "local_y_m", "reference_x_m", "reference_y_m"]
        ].reindex(cycles)
        numeric = numeric.interpolate(method="linear", limit_direction="both")
        positions[:, robot, :] = numeric[["local_x_m", "local_y_m"]].to_numpy()
        references[:, robot, :] = numeric[
            ["reference_x_m", "reference_y_m"]
        ].to_numpy()
    return {
        "cycles": cycles,
        "time_s": cycles * 0.2,
        "positions": positions,
        "references": references,
        "observed": observed,
    }


def reconstruct_connectivity(aligned: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    config = current_preflight_config()
    positions = aligned["positions"]
    references = aligned["references"]
    actual = np.empty(len(positions))
    reference = np.empty(len(positions))
    distances = np.empty((len(positions), len(PAIR_LIST)))
    for sample in range(len(positions)):
        _, actual_values, _ = exact_low_basis(
            positions[sample],
            PAIR_LIST,
            sigma=1.0,
            channel_parameters=config.channel,
        )
        _, reference_values, _ = exact_low_basis(
            references[sample],
            PAIR_LIST,
            sigma=1.0,
            channel_parameters=config.channel,
        )
        actual[sample] = actual_values[1]
        reference[sample] = reference_values[1]
    return {"actual": actual, "reference": reference, "distances": distances}


def architecture_robot_box(axis: plt.Axes, x: float, y: float, label: str) -> None:
    box = FancyBboxPatch(
        (x - 0.10, y - 0.055),
        0.20,
        0.11,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.85,
        edgecolor="#0072B2",
        facecolor=PALE_BLUE,
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(
        x,
        y + 0.020,
        label,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#0072B2",
        zorder=5,
    )
    axis.text(
        x,
        y - 0.020,
        "5-Hz loop",
        ha="center",
        va="center",
        fontsize=6.8,
        color=INK,
        zorder=5,
    )


def plot_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(2.65, 2.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    # 1. Top: Motion Capture / VRPN
    mocap = FancyBboxPatch(
        (0.10, 0.880),
        0.80,
        0.095,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.75,
        edgecolor="#2D6A4F",
        facecolor=PALE_GREEN,
        zorder=4,
    )
    axis.add_patch(mocap)
    axis.text(
        0.50,
        0.942,
        "Motion Capture / VRPN",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#1B4332",
        zorder=5,
    )
    axis.text(
        0.50,
        0.902,
        "own pose to each robot",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#2D6A4F",
        zorder=5,
    )

    # MoCap Bus Lines
    axis.plot([0.50, 0.50], [0.880, 0.852], color="#2D6A4F", lw=0.75, zorder=1)
    axis.plot([0.12, 0.88], [0.852, 0.852], color="#2D6A4F", lw=0.75, zorder=1)
    axis.annotate(
        "",
        xy=(0.12, 0.775),
        xytext=(0.12, 0.852),
        arrowprops=dict(arrowstyle="->", color="#2D6A4F", lw=0.75),
        zorder=2,
    )
    axis.annotate(
        "",
        xy=(0.88, 0.775),
        xytext=(0.88, 0.852),
        arrowprops=dict(arrowstyle="->", color="#2D6A4F", lw=0.75),
        zorder=2,
    )

    # 2. Four Robots
    coords = {
        "R3": (0.12, 0.715),
        "R2": (0.88, 0.715),
        "R4": (0.12, 0.285),
        "R1": (0.88, 0.285),
    }
    for label, (x, y) in coords.items():
        architecture_robot_box(axis, x, y, label)

    # 50 Hz Ring Edges (Blue with double arrows)
    axis.annotate(
        "",
        xy=(0.77, 0.760),
        xytext=(0.23, 0.760),
        arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.9),
        zorder=2,
    )
    axis.annotate(
        "",
        xy=(0.77, 0.240),
        xytext=(0.23, 0.240),
        arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.9),
        zorder=2,
    )
    axis.annotate(
        "",
        xy=(0.12, 0.350),
        xytext=(0.12, 0.650),
        arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.9),
        zorder=2,
    )
    axis.annotate(
        "",
        xy=(0.88, 0.350),
        xytext=(0.88, 0.650),
        arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.9),
        zorder=2,
    )

    # 3. Central Per-Robot 5-Hz Loop Box
    center = FancyBboxPatch(
        (0.26, 0.210),
        0.48,
        0.530,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.75,
        edgecolor="#94A3B8",
        facecolor="#F8FAFC",
        zorder=3,
    )
    axis.add_patch(center)
    axis.text(
        0.50,
        0.705,
        "Per-robot 5-Hz loop",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        zorder=5,
    )

    # 3 Steps in Central Box
    steps = [
        (0.635, "latest-value cache"),
        (0.540, "range / freshness gate"),
        (0.445, "local SOCP + guard"),
    ]
    for ys, ts in steps:
        sp = FancyBboxPatch(
            (0.285, ys - 0.032),
            0.43,
            0.064,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.6,
            edgecolor="#CBD5E1",
            facecolor="white",
            zorder=4,
        )
        axis.add_patch(sp)
        axis.text(
            0.50,
            ys,
            ts,
            ha="center",
            va="center",
            fontsize=6.8,
            color=INK,
            zorder=5,
        )

    # Arrows between steps
    for yf, yt in [(0.603, 0.572), (0.508, 0.477)]:
        axis.annotate(
            "",
            xy=(0.50, yt),
            xytext=(0.50, yf),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.6),
            zorder=5,
        )

    axis.text(
        0.50,
        0.368,
        "50-Hz peer state broadcast\nlatest-value; no barrier",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#0072B2",
        fontstyle="italic",
        zorder=5,
    )
    axis.text(
        0.50,
        0.260,
        "(executed independently\non each robot)",
        ha="center",
        va="center",
        fontsize=5.3,
        color=MUTED,
        zorder=5,
    )

    # 4. Bottom: Supervisor
    sup = FancyBboxPatch(
        (0.10, 0.012),
        0.80,
        0.125,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.75,
        edgecolor="#D55E00",
        facecolor=PALE_GOLD,
        zorder=4,
    )
    axis.add_patch(sup)
    axis.text(
        0.50,
        0.098,
        "Supervisor",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#B45309",
        zorder=5,
    )
    axis.text(
        0.50,
        0.064,
        "lifecycle, safety, and logging only",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#92400E",
        zorder=5,
    )
    axis.text(
        0.50,
        0.033,
        "(no per-cycle ticks or centralized solve)",
        ha="center",
        va="center",
        fontsize=5.6,
        color="#B45309",
        fontstyle="italic",
        zorder=5,
    )

    # Supervisor dashed management line to R4 and R1
    axis.plot([0.50, 0.50], [0.137, 0.155], color="#D55E00", ls="--", lw=0.75, zorder=1)
    axis.plot([0.12, 0.88], [0.155, 0.155], color="#D55E00", ls="--", lw=0.75, zorder=1)
    axis.annotate(
        "",
        xy=(0.12, 0.225),
        xytext=(0.12, 0.155),
        arrowprops=dict(arrowstyle="<->", color="#D55E00", ls="--", lw=0.75),
        zorder=2,
    )
    axis.annotate(
        "",
        xy=(0.88, 0.225),
        xytext=(0.88, 0.155),
        arrowprops=dict(arrowstyle="<->", color="#D55E00", ls="--", lw=0.75),
        zorder=2,
    )

    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(figure, output_dir, "physical_async_architecture")


def plot_trajectory(aligned: dict[str, np.ndarray], output_dir: Path) -> None:
    positions = aligned["positions"]
    references = aligned["references"]
    figure, axis = plt.subplots(figsize=(3.45, 2.55))
    for robot, color in enumerate(ROBOT_COLORS):
        axis.plot(
            references[:, robot, 0],
            references[:, robot, 1],
            color=color,
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.65,
        )
        axis.plot(
            positions[:, robot, 0],
            positions[:, robot, 1],
            color=color,
            linewidth=1.1,
            label=f"robot {robot + 1}",
        )
        axis.plot(
            positions[0, robot, 0],
            positions[0, robot, 1],
            marker="o",
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor=color,
        )
        axis.plot(
            positions[-1, robot, 0],
            positions[-1, robot, 1],
            marker="s",
            markersize=3.4,
            color=color,
        )
    for i, j in PAIR_LIST[:4]:
        axis.plot(
            [positions[-1, i, 0], positions[-1, j, 0]],
            [positions[-1, i, 1], positions[-1, j, 1]],
            color=MUTED,
            linewidth=0.6,
            alpha=0.65,
            zorder=0,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"local $x$ (m)")
    axis.set_ylabel(r"local $y$ (m)")
    axis.legend(loc="upper center", ncol=2, frameon=False, columnspacing=0.7)
    figure.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.99)
    save(figure, output_dir, "physical_async_trajectory")


def plot_connectivity(
    aligned: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    summary: dict,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(3.45, 2.55))
    time_s = aligned["time_s"]
    axis.plot(time_s, diagnostics["actual"], color="#0072B2", label="measured-state reconstruction")
    axis.plot(
        time_s,
        diagnostics["reference"],
        color="#D55E00",
        linestyle=(0, (4, 2)),
        label="reference state",
    )
    axis.axhline(0.50, color=INK, linestyle=(0, (2, 2)), linewidth=0.85, label=r"$\lambda_{\rm req}=0.50$")
    axis.text(
        0.03,
        0.06,
        rf"supervisor min. $\lambda_2={summary['minimum_diagnostic_lambda2']:.3f}$",
        transform=axis.transAxes,
        fontsize=6.8,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GRID),
    )
    axis.set_xlabel("local cycle time (s)")
    axis.set_ylabel(r"algebraic connectivity $\lambda_2$")
    axis.legend(loc="upper right", frameon=False)
    figure.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.99)
    save(figure, output_dir, "physical_async_connectivity")


def summarize(
    frame: pd.DataFrame,
    summary: dict,
    events: pd.DataFrame,
    aligned: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> dict:
    tracking = np.hypot(
        frame["local_x_m"] - frame["reference_x_m"],
        frame["local_y_m"] - frame["reference_y_m"],
    )
    neighbor_sets = sorted(frame["neighbor_ids"].unique().tolist())
    return {
        "agent_samples": summary["agent_samples"],
        "run_result": summary["result"],
        "run_reason": summary["reason"],
        "minimum_supervisor_lambda2": summary["minimum_diagnostic_lambda2"],
        "minimum_observed_separation_m": summary["minimum_observed_separation_m"],
        "minimum_observed_geofence_margin_m": summary[
            "minimum_observed_geofence_margin_m"
        ],
        "tracking_rmse_m": float(np.sqrt(np.mean(tracking**2))),
        "maximum_tracking_error_m": float(np.max(tracking)),
        "maximum_local_cycle_wall_time_ms": float(frame["cycle_wall_time_ms"].max()),
        "minimum_reported_local_soc_margin": float(frame["local_soc_margin"].min()),
        "local_degradation_reports": int((events["event"] == "local_degraded").sum()),
        "retained_neighbor_sets": neighbor_sets,
        "missing_cycle_reports_interpolated_for_figures": int(
            aligned["observed"].size - np.count_nonzero(aligned["observed"])
        ),
        "minimum_cycle_aligned_reconstructed_lambda2": float(
            diagnostics["actual"].min()
        ),
        "minimum_cycle_aligned_reference_lambda2": float(
            diagnostics["reference"].min()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
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
            / "async_metrics.json"
        ),
    )
    parser.add_argument(
        "--regenerate-architecture",
        action="store_true",
        help=(
            "regenerate the fallback Matplotlib architecture panel; by default "
            "preserve the manually authored publication figure"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, config, summary, events = load_run(args.run_dir)
    aligned = cycle_aligned(frame)
    diagnostics = reconstruct_connectivity(aligned)
    plot_architecture(args.output_dir)
    plot_trajectory(aligned, args.output_dir)
    plot_connectivity(aligned, diagnostics, summary, args.output_dir)
    metrics = summarize(frame, summary, events, aligned, diagnostics)
    payload = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "source_run": str(args.run_dir.resolve()),
        "source_sha256": {
            name: sha256(args.run_dir / name)
            for name in ("timeseries.csv", "config.json", "summary.json", "events.csv")
        },
        "implementation_claim": config["implementation_claim"],
        "metrics": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
