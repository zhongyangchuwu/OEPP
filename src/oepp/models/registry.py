"""Explicit direct-model registry for extensible OEPP architectures."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from oepp.data import FeatureKind, feature_dimension

from .attention import TransformerEncoder
from .kepp_adapted import AdaptedKEPP
from .mlp import MLP
from .p3iv_adapted import AdaptedP3IV

DIRECT_MODEL_FAMILIES = frozenset({"mlp", "transformer", "adapted_p3iv", "adapted_kepp"})


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
    if family == "adapted_kepp":
        graph_context = config.get("kepp_graph")
        if not isinstance(graph_context, Mapping):
            raise ValueError("adapted_kepp requires a prepared train-only graph context")
        node_embeddings = graph_context.get("node_embeddings")
        normalized_adjacency = graph_context.get("normalized_adjacency")
        if not isinstance(node_embeddings, torch.Tensor) or not isinstance(
            normalized_adjacency, torch.Tensor
        ):
            raise ValueError("adapted_kepp graph context must contain graph tensors")
        return AdaptedKEPP(
            dimension=dimension,
            horizon=horizon,
            hidden_dimension=int(model_config["hidden_dimension"]),
            num_heads=int(model_config["num_heads"]),
            graph_layers=int(model_config["graph_layers"]),
            dropout=float(model_config["dropout"]),
            node_embeddings=node_embeddings,
            normalized_adjacency=normalized_adjacency,
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
