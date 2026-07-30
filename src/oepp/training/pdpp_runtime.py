"""Checkpoint and reconstruction helpers shared by fresh PDPP training and embedding export."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from oepp.models import diffusion, temporal
from oepp.training.support import utc_now

PDPP_CHECKPOINT_FORMAT = "oepp-pdpp-embedding-v1"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_pdpp_diffusion(args: Any) -> torch.nn.Module:
    temporal_model = temporal.TemporalUnet(
        args,
        args.action_dim + args.observation_dim + args.class_dim + args.horizon_dim,
        dim=256,
        dim_mults=(1, 2, 4),
    )
    return diffusion.GaussianDiffusion(
        temporal_model,
        args.horizon,
        args.observation_dim,
        args.action_dim,
        args.horizon_dim,
        args.class_dim,
        args.n_diffusion_steps,
        loss_type="Weighted_MSE",
        clip_denoised=True,
    )


def checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    args: Any,
    validation_metrics: Mapping[str, float],
    tb_logdir: str,
) -> dict[str, Any]:
    return {
        "format": PDPP_CHECKPOINT_FORMAT,
        "created_at": utc_now(),
        "epoch": int(epoch),
        "model": {
            key: value.detach().cpu() for key, value in unwrap_model(model).state_dict().items()
        },
        "ema": {
            key: value.detach().cpu() for key, value in unwrap_model(ema_model).state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "args": vars(args).copy(),
        "validation_metrics": dict(validation_metrics),
        "tb_logdir": tb_logdir,
    }


def save_pdpp_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def load_pdpp_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != PDPP_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a fresh {PDPP_CHECKPOINT_FORMAT} checkpoint")
    required = {"epoch", "ema", "args", "validation_metrics"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"PDPP checkpoint is missing required keys: {sorted(missing)}")
    return checkpoint
