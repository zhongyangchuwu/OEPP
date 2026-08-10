"""Deterministic Random and Matching baselines for OEPP planning."""

from __future__ import annotations

import hashlib
import json
import random

import torch
import torch.nn.functional as functional

SIMPLE_BASELINE_PROTOCOL = "oepp-two-split-two-horizon-baselines-v1"


def random_action_indices(
    *,
    pool_size: int,
    horizon: int,
    split_id: str,
    partition: str,
    sample_id: str,
    seed: int = 42,
) -> list[int]:
    """Select one reproducible ordered action sequence without replacement."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if pool_size < horizon:
        raise ValueError("candidate pool must contain at least horizon actions")
    identity = json.dumps(
        [SIMPLE_BASELINE_PROTOCOL, seed, split_id, partition, horizon, sample_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "big")
    return random.Random(derived_seed).sample(range(pool_size), horizon)


def matching_action_indices(
    start_frames: torch.Tensor,
    goal_frames: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    horizon: int,
) -> torch.Tensor:
    """Decode the paper's training-free endpoint/intermediate matching baseline."""
    if horizon < 2:
        raise ValueError("matching baseline requires horizon >= 2")
    if start_frames.ndim != 3 or goal_frames.shape != start_frames.shape:
        raise ValueError("endpoint tensors must share shape [batch, observations, dimension]")
    if candidate_embeddings.ndim != 2:
        raise ValueError("candidate embeddings must have shape [candidates, dimension]")
    if start_frames.shape[-1] != candidate_embeddings.shape[-1]:
        raise ValueError("endpoint and candidate embedding dimensions must match")
    if candidate_embeddings.shape[0] < horizon:
        raise ValueError("candidate pool must contain at least horizon actions")

    start = functional.normalize(start_frames.mean(dim=1), dim=-1)
    goal = functional.normalize(goal_frames.mean(dim=1), dim=-1)
    candidates = functional.normalize(candidate_embeddings, dim=-1)
    start_scores = start @ candidates.T
    goal_scores = goal @ candidates.T
    first = start_scores.argmax(dim=-1)
    last = goal_scores.argmax(dim=-1)

    if horizon == 2:
        return torch.stack((first, last), dim=1)

    midpoint = functional.normalize((start + goal) / 2.0, dim=-1)
    midpoint_scores = midpoint @ candidates.T
    ranked = torch.argsort(midpoint_scores, dim=-1, descending=True, stable=True)
    intermediate = ranked[:, : horizon - 2]
    return torch.cat((first[:, None], intermediate, last[:, None]), dim=1)
