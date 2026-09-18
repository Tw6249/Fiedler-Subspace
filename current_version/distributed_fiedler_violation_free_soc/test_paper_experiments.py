#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from run_paper_experiments import (
    run_rectangle_crossing,
    run_repeated_root_recovery,
    simulate_topology_formation,
)
from violation_free_soc_core import MoxChannelParameters


class PaperEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channel = MoxChannelParameters()

    def test_crossing_study_contains_declared_comparators(self) -> None:
        data, summary = run_rectangle_crossing(
            total_steps=90,
            seeds=1,
            consensus_rounds=40,
            channel=self.channel,
        )
        expected = {
            "ours-r2-block",
            "ours-r1-ablation",
            "yang-2010-single-fiedler",
            "zhang-2022-scalar-monitor",
        }
        self.assertEqual(set(summary["method"]), expected)
        self.assertTrue(np.isfinite(data["lambda2_estimate_mean"]).all())
        rmse = summary.set_index("method")["lambda2_RMSE_mean"]
        self.assertLess(rmse["ours-r2-block"], rmse["ours-r1-ablation"])
        self.assertLess(float(data["fiedler_gap"].min()), 1e-12)

    def test_nonconverged_allocation_stays_safe_when_nominal_disconnects(self) -> None:
        nominal, nominal_trajectory = simulate_topology_formation(
            "nominal", total_steps=240, channel=self.channel
        )
        proposed, _ = simulate_topology_formation(
            "allocation", total_steps=240, channel=self.channel
        )
        self.assertLess(float(nominal["connectivity_safety_margin"].min()), -1e-3)
        self.assertGreaterEqual(
            float(proposed["connectivity_safety_margin"].min()), -1e-9
        )
        self.assertGreaterEqual(
            float(proposed["exact_global_matrix_margin"].min()), -1e-9
        )
        self.assertGreater(float(proposed["master_residual_eta_q"].min()), 1.0)
        self.assertLess(abs(float(nominal["lambda2_post"].iloc[-1])), 1e-12)
        self.assertLess(int(nominal["active_edge_count"].iloc[-1]), 5)
        self.assertGreater(float(proposed["lambda2_post"].iloc[-1]), 0.195)
        self.assertEqual(int(proposed["active_edge_count"].iloc[-1]), 9)
        terminal_distances = np.linalg.norm(
            nominal_trajectory[-1, 1:] - nominal_trajectory[-1, 0], axis=1
        )
        self.assertTrue(np.all(terminal_distances >= self.channel.cutoff_distance))

    def test_repeated_root_reports_positive_local_slater_diagnostic(self) -> None:
        _, summary = run_repeated_root_recovery(
            iterations=5, channel=self.channel
        )
        metrics = dict(zip(summary["metric"], summary["value"]))
        self.assertGreater(
            metrics["minimum local Slater diagnostic gamma_i_star"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
