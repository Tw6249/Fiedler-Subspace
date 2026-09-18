#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the paper's empirical estimator--controller comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


COLORS = {
    "exact": "#0072B2",
    "estimated": "#D55E00",
    "tightened": "#009E73",
    "unsafe": "#CC79A7",
    "neutral": "#6B7280",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.60,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "axes.grid": True,
            "grid.alpha": 0.14,
            "grid.linewidth": 0.32,
            "lines.linewidth": 1.00,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
        }
    )


def panel_caption(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        0.5,
        -0.30,
        rf"$\bf{{({label})}}$ {title}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color="#20252A",
        clip_on=False,
    )


def save_caption_free_panels(
    figure: plt.Figure,
    axes: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """Export the three axes separately for LaTeX ``subfig`` assembly."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for label, axis in zip(("a", "b", "c"), axes):
        display_bbox = Bbox.union([axis.get_tightbbox(renderer)])
        inches_bbox = display_bbox.transformed(figure.dpi_scale_trans.inverted()).padded(0.025)
        for extension, dpi in (("pdf", None), ("png", 300)):
            figure.savefig(
                output_dir / f"{stem}_{label}.{extension}",
                dpi=dpi,
                bbox_inches=inches_bbox,
                pad_inches=0.0,
                facecolor="white",
            )


def plot_end_to_end(
    exact: pd.DataFrame,
    estimated: pd.DataFrame,
    tightened: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_style()
    exact = exact[exact["method"] == "allocation"].reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.72))

    series = [
        ("Exact subspace", exact, COLORS["exact"], "-", 1.00),
        ("Estimated", estimated, COLORS["estimated"], "--", 0.90),
        (r"Estimated + $\sum_i\beta_i=0.003$", tightened, COLORS["tightened"], "-.", 1.18),
    ]

    axis = axes[0]
    for label, data, color, linestyle, linewidth in series:
        axis.plot(
            data["step"],
            data["connectivity_safety_margin"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.55)
    axis.set(xlabel="sampling instant", ylabel=r"$\lambda_2-\lambda_{\rm req}$")
    axis.legend(loc="upper right", ncol=1, fontsize=6.6)

    axis = axes[1]
    for label, data, color, linestyle, linewidth in series:
        axis.plot(
            data["step"],
            data["exact_global_matrix_margin"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.55)
    axis.set_yscale("symlog", linthresh=1e-3, linscale=0.8)
    axis.set(xlabel="sampling instant", ylabel=r"$\lambda_{\min}(\mathcal{M}_{\rm exact})$")

    axis = axes[2]
    axis.plot(
        estimated["step"],
        estimated["subspace_sine_error"],
        color=COLORS["estimated"],
        linestyle="--",
        linewidth=0.90,
        label="Estimated",
    )
    axis.plot(
        tightened["step"],
        tightened["subspace_sine_error"],
        color=COLORS["tightened"],
        linestyle="-.",
        linewidth=1.18,
        label=r"Estimated + $\sum_i\beta_i=0.003$",
    )
    axis.set(xlabel="sampling instant", ylabel=r"$\|\sin\Theta(\widehat V,V)\|_2$")
    axis.legend(loc="lower right", fontsize=6.6)

    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.29, top=0.97, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_name = "end_to_end_pipeline"
    save_caption_free_panels(figure, axes, output_dir, stem_name)
    panel_caption(axes[0], "a", "Sampled-time connectivity margin")
    panel_caption(axes[1], "b", "A posteriori exact-model barrier")
    panel_caption(axes[2], "c", "Distributed subspace tracking")
    stem = output_dir / stem_name
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--exact",
        type=Path,
        default=root / "paper_data" / "core" / "topology_formation.csv",
    )
    parser.add_argument(
        "--estimated",
        type=Path,
        default=root / "paper_data" / "end_to_end" / "end_to_end_estimated.csv",
    )
    parser.add_argument(
        "--tightened",
        type=Path,
        default=root / "paper_data" / "end_to_end_tightened" / "end_to_end_estimated.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "paper_figures")
    args = parser.parse_args()

    plot_end_to_end(
        pd.read_csv(args.exact),
        pd.read_csv(args.estimated),
        pd.read_csv(args.tightened),
        args.output_dir,
    )
    print(f"Wrote line-weight-refined end-to-end figure to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
