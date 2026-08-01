"""OEPP-native P3IV-style memory planner with a unified semantic decoder.

This implementation is derived from the approved conceptual adaptation contract, not from
upstream P3IV source. It intentionally omits the upstream fixed-class Viterbi decoder so the
same embedding decoder can evaluate frozen Base and Novel candidate pools.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class AdaptedP3IV(nn.Module):
    """Generate action-text embedding plans from endpoint VideoCLIP observations.

    The model retains three P3IV-style ideas: endpoint-conditioned memory tokens, latent
    Gaussian generation, and an explicit diversity penalty. Its public prediction contract
    returns a list of per-step action embeddings, matching the OEPP direct planner interface.
    """

    primary_decoder = "semantic_embedding_cosine"

    def __init__(
        self,
        *,
        dimension: int,
        horizon: int,
        hidden_dimension: int,
        memory_slots: int,
        noise_dimension: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension <= 0 or horizon <= 0 or hidden_dimension <= 0:
            raise ValueError("dimension, horizon, and hidden_dimension must be positive")
        if memory_slots <= 0 or noise_dimension <= 0:
            raise ValueError("memory_slots and noise_dimension must be positive")
        if num_layers <= 0 or num_heads <= 0 or hidden_dimension % num_heads:
            raise ValueError("hidden_dimension must divide evenly across positive attention heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dimension = dimension
        self.horizon = horizon
        self.noise_dimension = noise_dimension
        self.memory_slots = memory_slots
        self.state_encoder = nn.Sequential(
            nn.Linear(6 * dimension, 4 * dimension),
            nn.ReLU(),
            nn.Linear(4 * dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.memory_tokens = nn.Parameter(torch.empty(memory_slots, hidden_dimension))
        self.step_queries = nn.Parameter(torch.empty(horizon, hidden_dimension))
        self.noise_projection = nn.Linear(noise_dimension, hidden_dimension, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dimension,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.memory_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_decoder = nn.Linear(hidden_dimension, dimension)
        nn.init.normal_(self.memory_tokens, std=0.02)
        nn.init.normal_(self.step_queries, std=0.02)

    def _validate_frames(self, frames: torch.Tensor) -> None:
        if frames.ndim != 2 or frames.shape[1] != 6 * self.dimension:
            raise ValueError(
                f"Expected endpoint frames [batch, {6 * self.dimension}], got {tuple(frames.shape)}"
            )
        if not frames.is_floating_point():
            raise ValueError("Endpoint frame embeddings must be floating point tensors")

    def _sample_latent(
        self, frames: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.randn(
            (frames.shape[0], self.noise_dimension),
            device=frames.device,
            dtype=frames.dtype,
            generator=generator,
        )

    def _predict(self, frames: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        self._validate_frames(frames)
        expected_latent = (frames.shape[0], self.noise_dimension)
        if tuple(latent.shape) != expected_latent:
            raise ValueError(f"Expected latent shape {expected_latent}, got {tuple(latent.shape)}")
        if latent.device != frames.device or latent.dtype != frames.dtype:
            raise ValueError("Latent and endpoint frame embeddings must share device and dtype")
        state = self.state_encoder(frames).unsqueeze(1)
        memory = self.memory_tokens.unsqueeze(0).expand(frames.shape[0], -1, -1)
        queries = self.step_queries.unsqueeze(0).expand(frames.shape[0], -1, -1)
        queries = queries + self.noise_projection(latent).unsqueeze(1)
        tokens = torch.cat((state, memory, queries), dim=1)
        encoded = self.memory_transformer(tokens)
        return self.action_decoder(encoded[:, 1 + self.memory_slots :, :])

    def predict_with_generator(
        self, frames: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> list[torch.Tensor]:
        """Return one stochastic embedding trajectory using an explicit generator when supplied."""
        latent = self._sample_latent(frames, generator)
        predicted = self._predict(frames, latent)
        return [predicted[:, step, :] for step in range(self.horizon)]

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        return self.predict_with_generator(frames)

    def predict_from_latent(self, frames: torch.Tensor, latent: torch.Tensor) -> list[torch.Tensor]:
        """Predict from an explicit latent tensor for deterministic tests and diagnostics."""
        predicted = self._predict(frames, latent)
        return [predicted[:, step, :] for step in range(self.horizon)]

    def diversity_penalty(
        self, frames: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Penalize latent samples that collapse to indistinguishable action trajectories."""
        first_latent = self._sample_latent(frames, generator)
        second_latent = self._sample_latent(frames, generator)
        first_prediction = self._predict(frames, first_latent)
        second_prediction = self._predict(frames, second_latent)
        latent_distance = (first_latent - second_latent).abs().mean().clamp_min(1e-6)
        prediction_distance = (first_prediction - second_prediction).abs().mean()
        return 1.0 / (prediction_distance / latent_distance + 1e-6)


def stack_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Validate and stack the public per-step embedding output contract."""
    if not outputs:
        raise ValueError("P3IV-style planner produced no action embeddings")
    first = outputs[0]
    if not all(output.shape == first.shape for output in outputs):
        raise ValueError("P3IV-style planner emitted inconsistent per-step embedding shapes")
    return torch.stack(list(outputs), dim=1)
