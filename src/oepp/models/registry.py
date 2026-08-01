"""Explicit direct-model registry for extensible OEPP architectures."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from oepp.data import FeatureKind, feature_dimension

from .attention import TransformerEncoder
from .mlp import MLP
from .p3iv_adapted import AdaptedP3IV

DIRECT_MODEL_FAMILIES = frozenset({"mlp", "transformer", "adapted_p3iv"})


def build_direct_model(config: Mapping[str, object], device: torch.device) -> torch.nn.Module:
    """Build one registered direct embedding planner from a validated config mapping."""
    feature = FeatureKind(str(config["feature"]))
    horizon = int(config["horizon"])
    if horizon <= 0:
        raise ValueError("model horizon must be positive")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("model config must be a mapping")
    family = str(model_config.get("family"))
    dimension = feature_dimension(feature)
    if family == "mlp":
        return MLP(dim=dimension, T=horizon).to(device)
    if family == "transformer":
        return TransformerEncoder(
            input_dim=dimension,
            hidden_dim=dimension,
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            T=horizon,
        ).to(device)
    if family == "adapted_p3iv":
        return AdaptedP3IV(
            dimension=dimension,
            horizon=horizon,
            hidden_dimension=int(model_config["hidden_dimension"]),
            memory_slots=int(model_config["memory_slots"]),
            noise_dimension=int(model_config["noise_dimension"]),
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
    raise ValueError(f"Unsupported direct OEPP model family: {family!r}")
