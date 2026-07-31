"""Shared, versioned Table V observation and annotation protocol primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_HORIZON = 4
SUPPORTED_HORIZONS = frozenset({3, 4})
IMAGE_OFFSETS = {
    "1+1": {"start": (0,), "goal": (0,)},
    "3+3": {"start": (0, 1, 2), "goal": (-2, -1, 0)},
}
FRAME_OFFSETS = IMAGE_OFFSETS["3+3"]
SHARED_CACHE_PROTOCOL_VERSION = "shared_cache_v2"


def require_horizon(horizon: int) -> int:
    if horizon not in SUPPORTED_HORIZONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_HORIZONS))
        raise ValueError(f"Table V horizon must be one of {supported}, received {horizon}")
    return horizon


def frame_offsets(image_setting: str) -> dict[str, tuple[int, ...]]:
    try:
        return IMAGE_OFFSETS[image_setting]
    except KeyError as error:
        supported = ", ".join(sorted(IMAGE_OFFSETS))
        raise ValueError(
            f"Table V image setting must be one of {supported}, received {image_setting}"
        ) from error


def image_count(image_setting: str) -> int:
    return len(frame_offsets(image_setting)["start"])


@dataclass(frozen=True)
class TableVFrameProtocol:
    """A versioned visual-input contract for one Table V horizon and image setting."""

    horizon: int
    image_setting: str

    def __post_init__(self) -> None:
        require_horizon(self.horizon)
        frame_offsets(self.image_setting)

    @property
    def identifier(self) -> str:
        image_tag = self.image_setting.replace("+", "x")
        return f"table_v_t{self.horizon}_{image_tag}_{SHARED_CACHE_PROTOCOL_VERSION}"

    @property
    def offsets(self) -> dict[str, tuple[int, ...]]:
        return frame_offsets(self.image_setting)


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
