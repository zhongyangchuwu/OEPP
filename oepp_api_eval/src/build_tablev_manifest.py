from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from common import load_json, load_jsonl, write_jsonl
from extract_tablev_frames import HORIZON, PROTOCOL_ID

SPLIT_FILES = {
    "base": "test_base_dataset_1.json",
    "novel": "novel_dataset_1.json",
}
POOL_FILES = {
    "base": "base_action_pool_1.json",
    "novel": "novel_action_pool_1.json",
}


def _window_key(
    vid: str, start: float, end: float, actions: list[str]
) -> tuple[str, float, float, tuple[str, ...]]:
    return vid, round(start, 6), round(end, 6), tuple(actions)


def _annotation_windows(record: dict[str, Any]) -> list[dict[str, Any]]:
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
    windows: list[dict[str, Any]] = []
    if len(actions) >= HORIZON:
        for start_step in range(len(actions) - HORIZON + 1):
            end_step = start_step + HORIZON - 1
            windows.append(
                {
                    "start_step": start_step,
                    "end_step": end_step,
                    "start_f": float(segments[start_step][0]),
                    "end_f": float(segments[end_step][1]),
                    "action_list": actions[start_step : end_step + 1],
                }
            )
    else:
        windows.append(
            {
                "start_step": 0,
                "end_step": len(actions) - 1,
                "start_f": float(segments[0][0]),
                "end_f": float(segments[-1][1]),
                "action_list": [actions[0]] * (HORIZON - len(actions)) + actions,
            }
        )
    return windows


def _window_index(
    records: list[dict[str, Any]],
) -> dict[tuple[str, float, float, tuple[str, ...]], list[dict[str, Any]]]:
    index: dict[tuple[str, float, float, tuple[str, ...]], list[dict[str, Any]]] = {}
    for record in records:
        for window in _annotation_windows(record):
            key = _window_key(
                record["vid"], window["start_f"], window["end_f"], window["action_list"]
            )
            index.setdefault(key, []).append({**record, **window})
    return index


def _resolved_image_paths(observation: dict[str, Any], frame_root: Path, key: str) -> list[str]:
    relative_paths = observation.get(key)
    if not isinstance(relative_paths, list) or len(relative_paths) != 3:
        raise ValueError(f"{observation.get('sample_id')} must contain exactly three {key}")
    root = frame_root.resolve()
    resolved_paths: list[str] = []
    for relative_path in relative_paths:
        if not isinstance(relative_path, str):
            raise ValueError(f"{observation.get('sample_id')} has a non-string {key} path")
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents:
            raise ValueError(f"{observation.get('sample_id')} {key} escapes --frame-root")
        if not candidate.is_file():
            raise FileNotFoundError(f"{observation.get('sample_id')} missing image: {candidate}")
        try:
            with Image.open(candidate) as image:
                image.verify()
        except Exception as error:
            raise ValueError(
                f"{observation.get('sample_id')} unreadable image {candidate}: {error}"
            ) from error
        resolved_paths.append(str(candidate))
    return resolved_paths


def _candidate_actions(pool: list[str]) -> list[dict[str, Any]]:
    if not all(isinstance(action, str) and action for action in pool):
        raise ValueError("action pool must contain non-empty strings")
    if len(pool) != len(set(pool)):
        raise ValueError("action pool contains duplicate strings")
    return [{"id": index, "text": action} for index, action in enumerate(pool)]


def build_manifest(
    observations: list[dict[str, Any]],
    records: list[dict[str, Any]],
    action_pool: list[str],
    frame_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    windows = _window_index(records)
    candidates = _candidate_actions(action_pool)
    action_to_id = {candidate["text"]: candidate["id"] for candidate in candidates}
    manifest: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("observation index must contain JSON objects")
        if observation.get("protocol") != PROTOCOL_ID:
            raise ValueError(f"{observation.get('sample_id')} does not use protocol {PROTOCOL_ID}")
        if observation.get("split") != split or observation.get("image_setting") != "3+3":
            raise ValueError(
                f"{observation.get('sample_id')} has an incompatible split or image setting"
            )
        sample_id = observation.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("observation sample_id values must be unique non-empty strings")
        sample_ids.add(sample_id)
        action_list = observation.get("action_list")
        if (
            not isinstance(action_list, list)
            or len(action_list) != HORIZON
            or not all(isinstance(action, str) for action in action_list)
        ):
            raise ValueError(f"{sample_id} must contain exactly {HORIZON} action strings")
        if any(action not in action_to_id for action in action_list):
            raise ValueError(f"{sample_id} ground truth is outside the {split} action pool")
        vid, start_f, end_f = (
            observation.get("vid"),
            observation.get("start_f"),
            observation.get("end_f"),
        )
        if (
            not isinstance(vid, str)
            or not isinstance(start_f, (int, float))
            or not isinstance(end_f, (int, float))
        ):
            raise ValueError(f"{sample_id} has invalid sequence identity")
        matched = windows.get(_window_key(vid, float(start_f), float(end_f), action_list), [])
        if len(matched) != 1:
            raise ValueError(
                f"{sample_id} matches {len(matched)} annotation windows; expected exactly one"
            )
        source = matched[0]
        manifest.append(
            {
                "sample_id": sample_id,
                "vid": source["vid"],
                "dataset": source["dataset"],
                "event": source["task_name"],
                "split": split,
                "T": HORIZON,
                "window_start_step": source["start_step"],
                "window_end_step": source["end_step"],
                "image_setting": "3+3",
                "start_images": _resolved_image_paths(observation, frame_root, "start_images"),
                "end_images": _resolved_image_paths(observation, frame_root, "end_images"),
                "candidate_actions": candidates,
                "gt_action_ids": [action_to_id[action] for action in action_list],
                "gt_actions": action_list,
                "pool_type": "split",
                "candidate_order": "original_file_order",
                "candidate_order_seed": None,
                "protocol": PROTOCOL_ID,
                "response_parser": "legacy_numbered_action_names",
                "source_sequence_index": observation["sequence_index"],
                "source_video_path": observation["source_video_path"],
                "frame_timestamps": observation["frame_timestamps"],
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local, validated manifest from extracted OEPP Table V T=4, 3+3 observations."
        )
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--split", choices=("base", "novel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = load_json(arguments.data_root / SPLIT_FILES[arguments.split])
    action_pool = load_json(arguments.data_root / POOL_FILES[arguments.split])
    if not isinstance(records, list) or not isinstance(action_pool, list):
        parser.error("split annotations and action pool must be JSON lists")
    manifest = build_manifest(
        load_jsonl(arguments.observations),
        records,
        action_pool,
        arguments.frame_root,
        arguments.split,
    )
    write_jsonl(arguments.output, manifest)
    print(f"Wrote {len(manifest)} records to {arguments.output}")


if __name__ == "__main__":
    main()
