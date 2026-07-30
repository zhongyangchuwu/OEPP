"""Canonical, reproducible locations for externally supplied video features."""

from __future__ import annotations

from pathlib import Path

from oepp.data.features import default_videoclip_root


def videoclip_root(explicit_root: Path | None = None) -> Path:
    """Resolve the legacy feature path through the canonical OEPP contract."""
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    return default_videoclip_root()


def videoclip_feature_path(dataset: str, vid: str, explicit_root: Path | None = None) -> Path:
    return videoclip_root(explicit_root) / f"{dataset}_{vid}.npy"
