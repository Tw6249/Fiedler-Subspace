#!/usr/bin/env python3
"""Regression tests for the only supported K=2 MuJoCo preflight."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_preflight.current_onboard_profile import (
    CURRENT_CONSENSUS_ROUNDS,
    CURRENT_CONTROL_RATE_HZ,
    CURRENT_ROUTER_ROUND_LATENCY_MS,
    average_communication_rounds_per_second,
    communication_round_breakdown,
    communication_round_service_rate_hz,
    communication_rounds_per_cycle,
)
from mujoco_preflight.distributed_stack import AgentNode
from mujoco_preflight.network_emulator import NeighborPacket, NetworkEmulator
from mujoco_preflight.run_current_onboard_preflight import (
    _quantized_delay_s,
    _run_current_condition,
    run_current_onboard_preflight,
)
from mujoco_preflight.scenario import (
    PreflightConfig,
    current_preflight_config,
    rectangle_reference,
)
from violation_free_soc_core import MoxChannelParameters, mox_inspired_rate_and_derivative


def test_xml_loads_with_expected_translational_plant() -> None:
    xml = Path(__file__).parent / "mujoco_preflight" / "scene_four_uav.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nq == 12
    assert model.nv == 12
    assert model.nu == 12


def test_current_profile_is_the_only_board_candidate() -> None:
    config = current_preflight_config()
    assert config.experiment_profile == "current_onboard_k02_100hz"
    assert config.candidate_topology == "cycle"
    assert config.samples == 200
    assert config.control_period_s == pytest.approx(0.2)
    assert config.physics_timestep_s == pytest.approx(0.005)
    assert config.consensus_rounds == 2
    assert config.emulated_round_latency_ms == pytest.approx(10.0)
    assert config.allocation_updates == 1
    assert config.record_node_ritz_estimates


def test_current_communication_accounting() -> None:
    assert CURRENT_CONSENSUS_ROUNDS == 2
    assert CURRENT_CONTROL_RATE_HZ == pytest.approx(5.0)
    assert CURRENT_ROUTER_ROUND_LATENCY_MS == pytest.approx(10.0)
    assert communication_round_breakdown() == {
        "estimator_diffusion": 1,
        "mean_consensus": 2,
        "gram_consensus": 2,
        "procrustes_consensus": 2,
        "neighbor_model_packet": 1,
        "node_ritz_diagnostic": 2,
        "allocation_resource": 1,
        "allocation_dual_residual": 1,
    }
    assert communication_rounds_per_cycle() == 12
    assert average_communication_rounds_per_second() == pytest.approx(60.0)
    assert communication_round_service_rate_hz() == pytest.approx(100.0)
    assert communication_rounds_per_cycle(record_node_ritz_estimates=False) == 10


def test_reference_and_arena_match_the_physical_plan() -> None:
    config = current_preflight_config()
    crossing = rectangle_reference(config.crossing_time_s, config)
    assert np.ptp(crossing[:, 0]) == pytest.approx(np.ptp(crossing[:, 1]))
    np.testing.assert_allclose(
        rectangle_reference(config.duration_s, config),
        config.final_task_targets,
        atol=1e-12,
    )
    assert np.all(np.abs(config.initial_positions) < config.geofence_half_extents_m)
    assert np.all(np.abs(config.final_task_targets) < config.geofence_half_extents_m)


def test_scaled_channel_preserves_baseline_weight_profile() -> None:
    config = current_preflight_config()
    baseline = MoxChannelParameters(transition_distance=2.0, cutoff_distance=2.65)
    scale = 0.8
    for baseline_distance in (1.0, 2.0, 2.3, 2.64):
        baseline_weight, baseline_derivative = mox_inspired_rate_and_derivative(
            baseline_distance, baseline
        )
        scaled_weight, scaled_derivative = mox_inspired_rate_and_derivative(
            scale * baseline_distance, config.channel
        )
        assert scaled_weight == pytest.approx(baseline_weight, abs=1e-12)
        assert scaled_derivative == pytest.approx(
            baseline_derivative / scale, abs=1e-11
        )


def test_router_masks_every_noncycle_edge() -> None:
    config = current_preflight_config()
    positions = 0.25 * config.initial_positions
    router = NetworkEmulator(
        positions,
        config.channel,
        round_latency_ms=config.emulated_round_latency_ms,
        message_header_bytes=config.message_header_bytes,
        candidate_edges=config.candidate_edges,
    )
    packets = router.neighbor_packets(np.arange(8, dtype=float).reshape(4, 2))
    assert [[packet.sender for packet in node] for node in packets] == [
        [1, 3],
        [0, 2],
        [1, 3],
        [2, 0],
    ]
    assert len(router.active_edges) == 4


def test_local_conic_model_uses_only_neighbor_packets() -> None:
    agent = AgentNode(
        index=0,
        basis_row=np.array([0.4, -0.2]),
        allocation_potential=np.zeros(3),
        dual_warm_start=np.zeros(3),
    )
    packet = NeighborPacket(
        sender=1,
        position=np.array([1.0, 0.0]),
        basis_row=np.array([-0.3, 0.7]),
        weight=0.8,
        weight_gradient=np.array([-0.4, 0.1]),
    )
    with_neighbor = agent.local_conic_terms(
        [packet], node_count=4, alpha=2.0, lambda_des=0.54, beta=0.0025
    )
    without_neighbor = agent.local_conic_terms(
        [], node_count=4, alpha=2.0, lambda_des=0.54, beta=0.0025
    )
    assert np.linalg.norm(with_neighbor[0]) > 0.0
    assert np.linalg.norm(without_neighbor[0]) == 0.0


def test_diffusion_selects_the_low_invariant_subspace_at_crossing() -> None:
    config = current_preflight_config()
    square = rectangle_reference(config.crossing_time_s, config)
    router = NetworkEmulator(
        square,
        config.channel,
        round_latency_ms=config.emulated_round_latency_ms,
        message_header_bytes=config.message_header_bytes,
        candidate_edges=config.candidate_edges,
    )
    values = np.linalg.eigvalsh(router.laplacian)
    low_factor = abs(1.0 - config.diffusion_step * values[1])
    high_factor = abs(1.0 - config.diffusion_step * values[-1])
    assert low_factor > high_factor


def test_router_delay_is_conservative_and_bounded_by_period() -> None:
    assert _quantized_delay_s(41.0, 0.2, 0.005) == pytest.approx(0.045)
    assert _quantized_delay_s(199.0, 0.2, 0.005) == pytest.approx(0.2)
    assert _quantized_delay_s(250.0, 0.2, 0.005) == pytest.approx(0.2)


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar.*:RuntimeWarning")
def test_short_current_run_saves_machine_readable_evidence(tmp_path: Path) -> None:
    config: PreflightConfig = replace(
        current_preflight_config(),
        duration_s=0.4,
        crossing_time_s=0.2,
        outward_start_s=0.3,
        estimator_warmup_steps=2,
    )
    output = tmp_path / "short_current"
    timeseries, summary = _run_current_condition(config, output)
    assert len(timeseries) == 2
    assert np.all(timeseries["communication_rounds"] == 12)
    assert summary.loc[0, "average_communication_rounds_per_second"] == 60.0
    for name in (
        "preflight_config.json",
        "preflight_summary.csv",
        "preflight_timeseries.csv",
        "mujoco_state_proposed.npz",
    ):
        assert (output / name).exists()


def test_current_runner_refuses_to_overwrite_results(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_current_onboard_preflight(existing)
