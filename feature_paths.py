"""Canonical, reproducible locations for externally supplied video features."""
from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOCLIP_ROOT = REPOSITORY_ROOT / "features" / "OEPP_videoclip"
VIDEOCLIP_ROOT_ENV = "OEPP_VIDEOCLIP_ROOT"


def videoclip_root(explicit_root: Path | None = None) -> Path:
    """Resolve the VideoCLIP feature root without depending on a machine-specific mount."""
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    configured_root = os.environ.get(VIDEOCLIP_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return DEFAULT_VIDEOCLIP_ROOT


def videoclip_feature_path(dataset: str, vid: str, explicit_root: Path | None = None) -> Path:
    return videoclip_root(explicit_root) / f"{dataset}_{vid}.npy"
