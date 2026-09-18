"""Canonical communication budget for the current onboard preflight candidate."""

from __future__ import annotations

CURRENT_CONSENSUS_ROUNDS = 2
CURRENT_ROUTER_ROUND_LATENCY_MS = 10.0
CURRENT_CONTROL_PERIOD_S = 0.20
CURRENT_CONTROL_RATE_HZ = 1.0 / CURRENT_CONTROL_PERIOD_S


def communication_round_breakdown(
    consensus_rounds: int = CURRENT_CONSENSUS_ROUNDS,
    *,
    record_node_ritz_estimates: bool = True,
) -> dict[str, int]:
    """Return synchronous neighbor rounds used by one high-level cycle.

    The scalar node-Ritz estimate is a diagnostic and can be disabled without
    changing the controller.  The frozen current candidate retains it so that
    estimator quality remains observable during preflight.
    """
    if consensus_rounds < 1:
        raise ValueError("consensus_rounds must be positive")
    return {
        "estimator_diffusion": 1,
        "mean_consensus": consensus_rounds,
        "gram_consensus": consensus_rounds,
        "procrustes_consensus": consensus_rounds,
        "neighbor_model_packet": 1,
        "node_ritz_diagnostic": (
            consensus_rounds if record_node_ritz_estimates else 0
        ),
        "allocation_resource": 1,
        "allocation_dual_residual": 1,
    }


def communication_rounds_per_cycle(
    consensus_rounds: int = CURRENT_CONSENSUS_ROUNDS,
    *,
    record_node_ritz_estimates: bool = True,
) -> int:
    """Return the total synchronous rounds in one control cycle."""
    return sum(
        communication_round_breakdown(
            consensus_rounds,
            record_node_ritz_estimates=record_node_ritz_estimates,
        ).values()
    )


def average_communication_rounds_per_second(
    consensus_rounds: int = CURRENT_CONSENSUS_ROUNDS,
    *,
    control_period_s: float = CURRENT_CONTROL_PERIOD_S,
    record_node_ritz_estimates: bool = True,
) -> float:
    """Return average round demand over simulated physical time."""
    if control_period_s <= 0.0:
        raise ValueError("control_period_s must be positive")
    return communication_rounds_per_cycle(
        consensus_rounds,
        record_node_ritz_estimates=record_node_ritz_estimates,
    ) / control_period_s


def communication_round_service_rate_hz(
    round_latency_ms: float = CURRENT_ROUTER_ROUND_LATENCY_MS,
) -> float:
    """Return the service rate corresponding to one complete round latency."""
    if round_latency_ms <= 0.0:
        raise ValueError("round_latency_ms must be positive")
    return 1e3 / round_latency_ms

