"""Feature protocols and machine-independent feature path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FeatureKind(str, Enum):
    VIDEOCLIP = "videoclip"
    S3D = "s3d"


FEATURE_DIMENSIONS = {FeatureKind.VIDEOCLIP: 768, FeatureKind.S3D: 512}
VIDEOCLIP_ROOT_ENV = "OEPP_VIDEOCLIP_ROOT"


@dataclass(frozen=True)
class FeatureRoots:
    """External feature locations needed by an OEPP run."""

    videoclip: Path
    coin_s3d: Path | None = None
    crosstask_s3d: Path | None = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_videoclip_root() -> Path:
    configured = os.environ.get(VIDEOCLIP_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "features" / "OEPP_videoclip"


def feature_dimension(feature: FeatureKind | str) -> int:
    try:
        return FEATURE_DIMENSIONS[FeatureKind(feature)]
    except ValueError as error:
        raise ValueError(f"Unsupported OEPP feature type: {feature!r}") from error


def resolve_feature_path(
    record: dict[str, object], feature: FeatureKind | str, roots: FeatureRoots
) -> Path:
    """Return the exact feature path for one validated annotation record."""
    feature_kind = FeatureKind(feature)
    dataset = str(record["dataset"])
    vid = str(record["vid"])
    if feature_kind is FeatureKind.VIDEOCLIP:
        return roots.videoclip / f"{dataset}_{vid}.npy"
    if dataset == "COIN":
        if roots.coin_s3d is None:
            raise ValueError("coin_s3d must be configured for S3D features")
        return roots.coin_s3d / f"{record['task_name']}_{record['task_id_old']}_{vid}.npy"
    if roots.crosstask_s3d is None:
        raise ValueError("crosstask_s3d must be configured for S3D features")
    return roots.crosstask_s3d / f"{record['task_id_old']}_{vid}.npy"
