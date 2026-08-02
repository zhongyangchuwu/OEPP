"""OEPP-native KEPP-style semantic graph planner.

This module implements the approved VideoCLIP-only adaptation contract. It is not copied from
or compatible with the upstream KEPP source: graph nodes and edges come only from the frozen
OEPP Base training partition, and Base/Novel decoding remains semantic and split-specific.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class AdaptedKEPP(nn.Module):
    """Plan action embeddings by attending endpoint queries to a train-only action graph."""

    primary_decoder = "semantic_embedding_cosine"

    def __init__(
        self,
        *,
        dimension: int,
        horizon: int,
        hidden_dimension: int,
        num_heads: int,
        graph_layers: int,
        dropout: float,
        node_embeddings: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        if dimension <= 0 or horizon <= 0 or hidden_dimension <= 0:
            raise ValueError("dimension, horizon, and hidden_dimension must be positive")
        if num_heads <= 0 or hidden_dimension % num_heads:
            raise ValueError("hidden_dimension must divide evenly across positive attention heads")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self._validate_graph(node_embeddings, normalized_adjacency, dimension)
        self.dimension = dimension
        self.horizon = horizon
        self.register_buffer("graph_node_embeddings", node_embeddings.detach().clone())
        self.register_buffer("normalized_adjacency", normalized_adjacency.detach().clone())
        self.state_encoder = nn.Sequential(
            nn.Linear(6 * dimension, 4 * dimension),
            nn.ReLU(),
            nn.Linear(4 * dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.step_queries = nn.Parameter(torch.empty(horizon, hidden_dimension))
        self.node_projection = nn.Linear(dimension, hidden_dimension)
        self.graph_projections = nn.ModuleList(
            nn.Linear(hidden_dimension, hidden_dimension) for _ in range(graph_layers)
        )
        self.graph_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dimension) for _ in range(graph_layers)
        )
        self.graph_attention = nn.MultiheadAttention(
            hidden_dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.step_attention = nn.MultiheadAttention(
            hidden_dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dimension)
        self.action_decoder = nn.Linear(hidden_dimension, dimension)
        nn.init.normal_(self.step_queries, std=0.02)

    @staticmethod
    def _validate_graph(
        node_embeddings: torch.Tensor, normalized_adjacency: torch.Tensor, dimension: int
    ) -> None:
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != dimension:
            raise ValueError(
                f"Expected graph node embeddings [actions, {dimension}], got "
                f"{tuple(node_embeddings.shape)}"
            )
        node_count = node_embeddings.shape[0]
        if node_count == 0:
            raise ValueError("KEPP graph must contain at least one Base action")
        if normalized_adjacency.shape != (node_count, node_count):
            raise ValueError(
                "Expected normalized adjacency shape "
                f"({node_count}, {node_count}), got {tuple(normalized_adjacency.shape)}"
            )
        if not node_embeddings.is_floating_point() or not normalized_adjacency.is_floating_point():
            raise ValueError("KEPP graph tensors must be floating point")
        nodes_are_finite = torch.isfinite(node_embeddings).all()
        adjacency_is_finite = torch.isfinite(normalized_adjacency).all()
        if not nodes_are_finite or not adjacency_is_finite:
            raise ValueError("KEPP graph tensors must be finite")
        if torch.any(normalized_adjacency < 0):
            raise ValueError("KEPP normalized adjacency must be non-negative")
        expected_rows = torch.ones(node_count, device=normalized_adjacency.device)
        if not torch.allclose(normalized_adjacency.sum(dim=1), expected_rows, atol=1e-6):
            raise ValueError("KEPP normalized adjacency rows must sum to one")

    def _validate_frames(self, frames: torch.Tensor) -> None:
        if frames.ndim != 2 or frames.shape[1] != 6 * self.dimension:
            raise ValueError(
                f"Expected endpoint frames [batch, {6 * self.dimension}], got {tuple(frames.shape)}"
            )
        if not frames.is_floating_point():
            raise ValueError("Endpoint frame embeddings must be floating point tensors")

    def _graph_node_states(self) -> torch.Tensor:
        nodes = self.node_projection(self.graph_node_embeddings)
        for projection, norm in zip(self.graph_projections, self.graph_norms):
            message = torch.matmul(self.normalized_adjacency, projection(nodes))
            nodes = norm(nodes + functional.gelu(message))
        return nodes

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        self._validate_frames(frames)
        state = self.state_encoder(frames).unsqueeze(1)
        queries = state + self.step_queries.unsqueeze(0)
        nodes = self._graph_node_states().unsqueeze(0).expand(frames.shape[0], -1, -1)
        graph_context, _ = self.graph_attention(queries, nodes, nodes, need_weights=False)
        fused = self.query_norm(queries + graph_context)
        planned, _ = self.step_attention(fused, fused, fused, need_weights=False)
        output = self.action_decoder(planned)
        return [output[:, step, :] for step in range(self.horizon)]
