"""Synchronous positive-weight neighbor router with packet accounting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from violation_free_soc_core import (
    MoxChannelParameters,
    TopologyEdge,
    complete_topology,
    graph_laplacian,
    mox_inspired_edges_and_gradients,
    state_induced_consensus_matrix,
)

Array = np.ndarray


@dataclass(frozen=True)
class NeighborPacket:
    """Information exposed to one receiver by one currently active neighbor."""

    sender: int
    position: Array
    basis_row: Array
    weight: float
    weight_gradient: Array


@dataclass
class CommunicationStats:
    node_count: int
    synchronous_rounds: int = 0
    directed_messages: int = 0
    bytes_sent_per_node: Array = field(init=False)

    def __post_init__(self) -> None:
        self.bytes_sent_per_node = np.zeros(self.node_count, dtype=np.int64)

    @property
    def total_bytes(self) -> int:
        return int(np.sum(self.bytes_sent_per_node))


class NetworkEmulator:
    """Route only active-edge messages; never create non-neighbor payloads."""

    def __init__(
        self,
        positions: Array,
        channel: MoxChannelParameters,
        *,
        round_latency_ms: float,
        message_header_bytes: int,
        candidate_edges: Sequence[TopologyEdge] | None = None,
        scalar_bytes: int = 8,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float).copy()
        self.channel = channel
        self.node_count = len(self.positions)
        self.round_latency_ms = float(round_latency_ms)
        self.message_header_bytes = int(message_header_bytes)
        self.scalar_bytes = int(scalar_bytes)
        pairs = (
            complete_topology(self.node_count)
            if candidate_edges is None
            else list(candidate_edges)
        )
        self.edges, self.gradients = mox_inspired_edges_and_gradients(
            self.positions,
            pairs,
            channel,
        )
        self.active_edges = [
            (i, j, weight) for i, j, weight in self.edges if weight > 0.0
        ]
        self.laplacian = graph_laplacian(self.node_count, self.edges)
        self.mixing, self.consensus_step = state_induced_consensus_matrix(
            self.laplacian,
            weighted_degree_bound=float(self.node_count - 1),
        )
        self.stats = CommunicationStats(self.node_count)

    def _record_rounds(self, payload_scalars: int, rounds: int = 1) -> None:
        if payload_scalars < 1 or rounds < 1:
            raise ValueError("payload size and round count must be positive")
        packet_bytes = self.message_header_bytes + self.scalar_bytes * payload_scalars
        for _ in range(rounds):
            self.stats.synchronous_rounds += 1
            for i, j, _ in self.active_edges:
                self.stats.directed_messages += 2
                self.stats.bytes_sent_per_node[i] += packet_bytes
                self.stats.bytes_sent_per_node[j] += packet_bytes

    def consensus(self, samples: Array, rounds: int) -> Array:
        """Execute same-graph synchronous neighbor consensus."""
        estimates = np.asarray(samples, dtype=float).copy()
        payload_scalars = int(np.prod(estimates.shape[1:]))
        self._record_rounds(payload_scalars, rounds)
        for _ in range(rounds):
            estimates = np.tensordot(self.mixing, estimates, axes=(1, 0))
        return estimates

    def laplacian_apply(self, values: Array) -> Array:
        """One neighbor exchange implementing the weighted graph Laplacian."""
        values = np.asarray(values, dtype=float)
        payload_scalars = int(np.prod(values.shape[1:]))
        self._record_rounds(payload_scalars, 1)
        return np.tensordot(self.laplacian, values, axes=(1, 0))

    def neighbor_packets(self, basis_rows: Array) -> list[list[NeighborPacket]]:
        """Deliver positions and estimator rows only along active edges."""
        basis_rows = np.asarray(basis_rows, dtype=float)
        # position (2), basis row (2), edge weight (1), and local gradient (2)
        self._record_rounds(payload_scalars=7, rounds=1)
        packets: list[list[NeighborPacket]] = [
            [] for _ in range(self.node_count)
        ]
        for i, j, weight in self.active_edges:
            packets[i].append(
                NeighborPacket(
                    sender=j,
                    position=self.positions[j].copy(),
                    basis_row=basis_rows[j].copy(),
                    weight=float(weight),
                    weight_gradient=self.gradients[(i, j)].copy(),
                )
            )
            packets[j].append(
                NeighborPacket(
                    sender=i,
                    position=self.positions[i].copy(),
                    basis_row=basis_rows[i].copy(),
                    weight=float(weight),
                    weight_gradient=self.gradients[(j, i)].copy(),
                )
            )
        return packets

    def virtual_communication_time_ms(self) -> float:
        return self.round_latency_ms * self.stats.synchronous_rounds

    def summary(self) -> dict[str, float | int]:
        return {
            "active_edge_count": len(self.active_edges),
            "communication_rounds": self.stats.synchronous_rounds,
            "directed_messages": self.stats.directed_messages,
            "communication_bytes": self.stats.total_bytes,
            "maximum_node_bytes": int(np.max(self.stats.bytes_sent_per_node)),
            "virtual_communication_time_ms": self.virtual_communication_time_ms(),
        }
