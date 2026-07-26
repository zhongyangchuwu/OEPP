"""Pure validation-only checkpoint selection rules for Experiment 4."""
from __future__ import annotations

from typing import Mapping, Optional


def is_better_direct_checkpoint(
    candidate: Mapping[str, float],
    best: Optional[Mapping[str, float]],
    epoch: int,
    best_epoch: Optional[int],
) -> bool:
    """Prefer validation SR, then accuracy, then lower MSE, then the earlier epoch."""
    if best is None or best_epoch is None:
        return True
    return (candidate["sr"], candidate["acc"], -candidate["mse"], -epoch) > (
        best["sr"],
        best["acc"],
        -best["mse"],
        -best_epoch,
    )
