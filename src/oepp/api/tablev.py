"""Shared, versioned Table V observation and annotation protocol primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_HORIZON = 4
SUPPORTED_HORIZONS = frozenset({3, 4})
FRAME_OFFSETS = {"start": (0, 1, 2), "goal": (-2, -1, 0)}


def require_horizon(horizon: int) -> int:
    if horizon not in SUPPORTED_HORIZONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_HORIZONS))
        raise ValueError(f"Table V horizon must be one of {supported}, received {horizon}")
    return horizon


def protocol_id(horizon: int) -> str:
    return f"table_v_t{require_horizon(horizon)}_3x3_legacy_v1"


def sample_id(split: str, index: int, record: dict[str, Any], horizon: int) -> str:
    require_horizon(horizon)
    return f"tablev_T{horizon}_{split}_{index:04d}_{record['vid']}"


def action_key(action: str) -> str:
    return "".join(action.casefold().split())


def window_key(
    vid: str, start: float, end: float, actions: Sequence[str]
) -> tuple[str, float, float, tuple[str, ...]]:
    return vid, round(start, 6), round(end, 6), tuple(actions)


def annotation_windows(record: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    require_horizon(horizon)
    annotations = record.get("anno")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"{record.get('vid', '<unknown>')} has no annotations")
    actions = [step.get("action") for step in annotations]
    if not all(isinstance(action, str) for action in actions):
        raise ValueError(f"{record.get('vid', '<unknown>')} annotations have invalid actions")
    segments = [step.get("segment") for step in annotations]
    if not all(
        isinstance(segment, list)
        and len(segment) == 2
        and all(isinstance(timestamp, (int, float)) for timestamp in segment)
        for segment in segments
    ):
        raise ValueError(f"{record.get('vid', '<unknown>')} annotations have invalid segments")
    if len(actions) >= horizon:
        return [
            {
                "start_step": start_step,
                "end_step": start_step + horizon - 1,
                "start_f": float(segments[start_step][0]),
                "end_f": float(segments[start_step + horizon - 1][1]),
                "action_list": actions[start_step : start_step + horizon],
            }
            for start_step in range(len(actions) - horizon + 1)
        ]
    return [
        {
            "start_step": 0,
            "end_step": len(actions) - 1,
            "start_f": float(segments[0][0]),
            "end_f": float(segments[-1][1]),
            "action_list": [actions[0]] * (horizon - len(actions)) + actions,
        }
    ]


def annotation_window_index(
    records: Sequence[dict[str, Any]], horizon: int
) -> dict[tuple[str, float, float, tuple[str, ...]], list[dict[str, Any]]]:
    index: dict[tuple[str, float, float, tuple[str, ...]], list[dict[str, Any]]] = {}
    for record in records:
        for window in annotation_windows(record, horizon):
            identity = window_key(
                record["vid"], window["start_f"], window["end_f"], window["action_list"]
            )
            index.setdefault(identity, []).append({**record, **window})
    return index
