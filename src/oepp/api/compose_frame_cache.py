"""Compose planned shared frames into validated Table V observations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import load_jsonl, sha256_file, write_jsonl
from .frame_cache import FRAME_CACHE_SCHEMA
from .tablev import TableVFrameProtocol, image_count


def _frame_outcomes_by_id(outcomes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        frame_id = outcome.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame extraction index has an invalid frame_id")
        if frame_id in indexed:
            raise ValueError(f"frame extraction index duplicates frame_id {frame_id}")
        indexed[frame_id] = outcome
    return indexed


def _cache_path(cache_root: Path, relative_path: str) -> Path:
    root = cache_root.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents:
        raise ValueError(f"cached frame path escapes --cache-root: {relative_path}")
    return candidate


def _resolve_frames(
    observation: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, Any]],
    cache_root: Path,
    role: str,
    expected_count: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    frame_ids = observation.get("frame_ids")
    if not isinstance(frame_ids, Mapping):
        raise ValueError(f"{observation.get('sample_id')} has no frame_ids mapping")
    role_ids = frame_ids.get(role)
    if not isinstance(role_ids, list) or len(role_ids) != expected_count:
        raise ValueError(f"{observation.get('sample_id')} has invalid {role} frame IDs")
    relative_paths: list[str] = []
    hashes: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for frame_id in role_ids:
        if not isinstance(frame_id, str):
            raise ValueError(f"{observation.get('sample_id')} has a non-string frame ID")
        outcome = outcomes.get(frame_id)
        if outcome is None:
            unavailable.append({"frame_id": frame_id, "reason": "missing_frame_outcome"})
            continue
        if outcome.get("status") != "available":
            unavailable.append(
                {
                    "frame_id": frame_id,
                    "reason": "unavailable_frame",
                    "error_type": outcome.get("error_type"),
                    "error": outcome.get("error"),
                }
            )
            continue
        relative_path = outcome.get("relative_path")
        digest = outcome.get("image_sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            unavailable.append({"frame_id": frame_id, "reason": "invalid_available_outcome"})
            continue
        path = _cache_path(cache_root, relative_path)
        if not path.is_file():
            unavailable.append({"frame_id": frame_id, "reason": "cached_file_missing"})
            continue
        if sha256_file(path) != digest:
            unavailable.append({"frame_id": frame_id, "reason": "cached_file_hash_mismatch"})
            continue
        relative_paths.append(relative_path)
        hashes.append(digest)
    return relative_paths, hashes, unavailable


def compose_observations(
    planned_observations: list[dict[str, Any]],
    frame_outcomes: list[dict[str, Any]],
    cache_root: Path,
    protocol: TableVFrameProtocol,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce manifest-ready observations, preserving unavailable frame dependencies."""
    if split not in {"base", "novel"}:
        raise ValueError(f"unsupported Table V split: {split!r}")
    outcomes = _frame_outcomes_by_id(frame_outcomes)
    available: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    expected_count = image_count(protocol.image_setting)
    for observation in planned_observations:
        if observation.get("protocol") != protocol.identifier or observation.get("split") != split:
            continue
        if observation.get("schema") != FRAME_CACHE_SCHEMA:
            raise ValueError(f"{observation.get('sample_id')} has the wrong cache schema")
        if observation.get("image_setting") != protocol.image_setting:
            raise ValueError(f"{observation.get('sample_id')} has the wrong image setting")
        start_paths, start_hashes, start_errors = _resolve_frames(
            observation, outcomes, cache_root, "start", expected_count
        )
        end_paths, end_hashes, end_errors = _resolve_frames(
            observation, outcomes, cache_root, "goal", expected_count
        )
        if start_errors or end_errors:
            unavailable.append(
                {
                    "schema": FRAME_CACHE_SCHEMA,
                    "cache_id": observation.get("cache_id"),
                    "protocol": protocol.identifier,
                    "split": split,
                    "sample_id": observation.get("sample_id"),
                    "status": "unavailable_observation",
                    "frame_errors": {"start": start_errors, "goal": end_errors},
                }
            )
            continue
        available.append(
            {
                "schema": FRAME_CACHE_SCHEMA,
                "cache_id": observation["cache_id"],
                "protocol": protocol.identifier,
                "split": split,
                "sample_id": observation["sample_id"],
                "sequence_index": observation["source_record_index"],
                "vid": observation["vid"],
                "source_video_path": observation["source_video_path"],
                "start_f": observation["start_f"],
                "end_f": observation["end_f"],
                "action_list": observation["action_list"],
                "image_setting": protocol.image_setting,
                "frame_timestamps": observation["frame_timestamps"],
                "start_images": start_paths,
                "end_images": end_paths,
                "frame_sha256": {"start": start_hashes, "goal": end_hashes},
            }
        )
    return available, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose one Table V protocol/split from a shared frame cache."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(3, 4), required=True)
    parser.add_argument("--image-setting", choices=("1+1", "3+3"), required=True)
    parser.add_argument("--split", choices=("base", "novel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unavailable-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.unavailable_output.exists():
        raise FileExistsError("refusing to overwrite composed observation outputs")
    protocol = TableVFrameProtocol(arguments.horizon, arguments.image_setting)
    available, unavailable = compose_observations(
        load_jsonl(arguments.observations),
        load_jsonl(arguments.frame_index),
        arguments.cache_root,
        protocol,
        arguments.split,
    )
    write_jsonl(arguments.output, available)
    write_jsonl(arguments.unavailable_output, unavailable)
    print(
        f"Composed {len(available)} observations; "
        f"recorded {len(unavailable)} unavailable observations."
    )


if __name__ == "__main__":
    main()
