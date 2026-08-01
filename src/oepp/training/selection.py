"""Pure validation-only checkpoint selection rules for Experiment 4."""

from __future__ import annotations

from collections.abc import Mapping


def is_better_direct_checkpoint(
    candidate: Mapping[str, float],
    best: Mapping[str, float] | None,
    epoch: int,
    best_epoch: int | None,
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


def is_better_pdpp_checkpoint(
    candidate: Mapping[str, float],
    best: Mapping[str, float] | None,
    epoch: int,
    best_epoch: int | None,
) -> bool:
    """Prefer validation SR, then accuracy, then the earlier epoch for PDPP."""
    if best is None or best_epoch is None:
        return True
    return (candidate["sr"], candidate["acc"], -epoch) > (
        best["sr"],
        best["acc"],
        -best_epoch,
    )
