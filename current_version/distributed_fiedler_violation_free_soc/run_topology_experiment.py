#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the paper's exact-subspace topology-disconnection stress test.

This entry point intentionally regenerates only the 240-sample controller
study used in Fig. 3.  The crossing and frozen-recovery records remain the
author-confirmed core data, while the end-to-end estimated-subspace study
continues to use the milder ``SIMPLE_GOAL`` target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_experiments import (
    TOPOLOGY_GOAL,
    TOPOLOGY_INITIAL,
    run_topology_formation,
)
from violation_free_soc_core import MoxChannelParameters


def scenario_table() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label, positions in (
        ("initial", TOPOLOGY_INITIAL),
        ("goal", TOPOLOGY_GOAL),
    ):
        for robot, (x_m, y_m) in enumerate(positions, start=1):
            rows.append(
                {
                    "configuration": label,
                    "robot": robot,
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_data" / "topology",
    )
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    channel = MoxChannelParameters()
    data, summary, trajectories = run_topology_formation(
        total_steps=args.steps,
        channel=channel,
    )
    data.to_csv(args.outdir / "topology_formation.csv", index=False)
    summary.to_csv(args.outdir / "topology_formation_summary.csv", index=False)
    scenario_table().to_csv(args.outdir / "topology_scenario.csv", index=False)
    np.savez_compressed(
        args.outdir / "topology_trajectories.npz",
        **trajectories,
    )

    print("\nTopology-formation summary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
