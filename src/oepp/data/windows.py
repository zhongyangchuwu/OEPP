"""Deterministic action-window construction independent of feature loading."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PaddingPolicy(str, Enum):
    NONE = "none"
    LEFT = "left"


@dataclass(frozen=True)
class SequenceWindow:
    sample_id: str
    split_id: str
    partition: str
    dataset: str
    vid: str
    task_name: str
    task_id: str
    task_id_old: str
    source_video_index: int
    start_step: int
    end_step: int
    is_padded: bool
    pad_count: int
    actions: tuple[str, ...]
    start_segment: tuple[float, float]
    end_segment: tuple[float, float]


def _annotation_action(step: dict[str, Any]) -> str:
    action = step.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("Each annotation step requires a non-empty action string")
    return action


def _segment(step: dict[str, Any]) -> tuple[float, float]:
    segment = step.get("segment")
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError("Each annotation step requires a two-value segment")
    return float(segment[0]), float(segment[1])


def build_windows(
    records: Iterable[dict[str, Any]],
    *,
    split_id: str,
    partition: str,
    horizon: int,
    padding: PaddingPolicy | str,
) -> list[SequenceWindow]:
    """Create stable, left-padded OEPP windows without mutating annotation records."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    padding_policy = PaddingPolicy(padding)
    windows: list[SequenceWindow] = []
    for source_video_index, record in enumerate(records):
        annotations = record.get("anno")
        if not isinstance(annotations, list) or not annotations:
            continue
        actions = [_annotation_action(step) for step in annotations]
        segments = [_segment(step) for step in annotations]
        dataset = str(record.get("dataset", ""))
        vid = str(record.get("vid", ""))
        if not dataset or not vid:
            raise ValueError("Each annotation record requires dataset and vid")
        common = {
            "split_id": split_id,
            "partition": partition,
            "dataset": dataset,
            "vid": vid,
            "task_name": str(record.get("task_name", "")),
            "task_id": str(record.get("task_id", "")),
            "task_id_old": str(record.get("task_id_old", "")),
            "source_video_index": source_video_index,
        }
        if len(actions) >= horizon:
            for start_step in range(len(actions) - horizon + 1):
                end_step = start_step + horizon - 1
                windows.append(
                    SequenceWindow(
                        sample_id=f"{split_id}_{partition}_{dataset}_{vid}_start{start_step}_T{horizon}",
                        start_step=start_step,
                        end_step=end_step,
                        is_padded=False,
                        pad_count=0,
                        actions=tuple(actions[start_step : end_step + 1]),
                        start_segment=segments[start_step],
                        end_segment=segments[end_step],
                        **common,
                    )
                )
        elif padding_policy is PaddingPolicy.LEFT:
            pad_count = horizon - len(actions)
            windows.append(
                SequenceWindow(
                    sample_id=f"{split_id}_{partition}_{dataset}_{vid}_padded_T{horizon}",
                    start_step=0,
                    end_step=len(actions) - 1,
                    is_padded=True,
                    pad_count=pad_count,
                    actions=tuple([actions[0]] * pad_count + actions),
                    start_segment=segments[0],
                    end_segment=segments[-1],
                    **common,
                )
            )
    return windows
