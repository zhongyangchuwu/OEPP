"""Shared reproducibility helpers for OEPP embedding experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import torch

from model.MLP import MLP
from model.attention import TransformerEncoder


DIRECT_CHECKPOINT_FORMAT = "oepp-direct-embedding-v1"
FEATURE_DIMS = {"s3d": 512, "videoclip": 768}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Union[str, Path], value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def feature_dimension(feature: str) -> int:
    try:
        return FEATURE_DIMS[feature]
    except KeyError as error:
        raise ValueError(f"Unsupported OEPP feature type: {feature}") from error


def load_action_embedding_dict(data_root: Union[str, Path], feature: str) -> dict[str, Any]:
    filename = {
        "s3d": "s3d_action_feat_dict.json",
        "videoclip": "vc_action_feat_dict.json",
    }.get(feature)
    if filename is None:
        raise ValueError(f"Unsupported OEPP feature type: {feature}")
    return json.loads((Path(data_root) / filename).read_text(encoding="utf-8"))


def load_action_pool(data_root: Union[str, Path], split: int, partition: str) -> list[str]:
    filename = {
        "base": f"base_action_pool_{split}.json",
        "novel": f"novel_action_pool_{split}.json",
        "total": "total_action_pool.json",
    }.get(partition)
    if filename is None:
        raise ValueError(f"Unsupported action-pool partition: {partition}")
    return json.loads((Path(data_root) / filename).read_text(encoding="utf-8"))


def action_embedding_tensor(
    actions: list[str], action_embedding_dict: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    try:
        vectors = [torch.as_tensor(action_embedding_dict[action], dtype=torch.float32) for action in actions]
    except KeyError as error:
        raise ValueError(f"Action embedding missing for {error.args[0]!r}") from error
    tensor = torch.cat(vectors, dim=0)
    return tensor.to(device)


def build_direct_model(config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    feature = str(config["feature"])
    dimension = feature_dimension(feature)
    horizon = int(config["T"])
    model_config = config["model"]
    model_name = model_config["model_n"]
    if model_name == "MLP":
        model = MLP(dim=dimension, T=horizon)
    elif model_name == "attention":
        model = TransformerEncoder(
            input_dim=dimension,
            hidden_dim=dimension,
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            T=horizon,
        )
    else:
        raise ValueError(f"Unsupported direct OEPP model: {model_name}")
    return model.to(device)


def stack_direct_outputs(outputs: list[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned no step embeddings")
    return torch.stack(outputs, dim=1)


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_direct_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    validation_metrics: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> None:
    payload = {
        "format": DIRECT_CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "model_state": state_dict_cpu(model),
        "optimizer_state": optimizer.state_dict(),
        "config": dict(config),
        "validation_metrics": dict(validation_metrics),
        "provenance": dict(provenance),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_direct_checkpoint(path: Union[str, Path], device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != DIRECT_CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {DIRECT_CHECKPOINT_FORMAT} checkpoint; "
            "do not load legacy pickled model objects for Experiment 4."
        )
    required = {"model_state", "config", "validation_metrics", "provenance"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    return checkpoint


def annotation_hashes(data_root: Union[str, Path], split: int, feature: str = "videoclip") -> dict[str, str]:
    root = Path(data_root)
    action_embedding_file = {
        "s3d": "s3d_action_feat_dict.json",
        "videoclip": "vc_action_feat_dict.json",
    }.get(feature)
    if action_embedding_file is None:
        raise ValueError(f"Unsupported OEPP feature type: {feature}")
    files = {
        "train": root / f"train_train_base_dataset_{split}.json",
        "validation": root / f"train_val_base_dataset_{split}.json",
        "base_test": root / f"test_base_dataset_{split}.json",
        "novel_test": root / f"novel_dataset_{split}.json",
        "action_embeddings": root / action_embedding_file,
    }
    return {name: sha256_file(path) for name, path in files.items()}
