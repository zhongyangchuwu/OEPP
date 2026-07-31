from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from oepp.data import ActionPool, Partition, SplitBundle

from .common import load_jsonl, write_jsonl
from .tablev import (
    DEFAULT_HORIZON,
    TableVFrameProtocol,
    annotation_window_index,
    image_count,
    protocol_id,
    require_horizon,
    window_key,
)


def _resolved_image_paths(
    observation: dict[str, Any], frame_root: Path, key: str, expected_count: int
) -> list[str]:
    relative_paths = observation.get(key)
    if not isinstance(relative_paths, list) or len(relative_paths) != expected_count:
        raise ValueError(
            f"{observation.get('sample_id')} must contain exactly {expected_count} {key}"
        )
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
    horizon: int = DEFAULT_HORIZON,
    image_setting: str = "3+3",
    expected_protocol: str | None = None,
) -> list[dict[str, Any]]:
    horizon = require_horizon(horizon)
    expected_count = image_count(image_setting)
    if expected_protocol is None:
        if image_setting != "3+3":
            raise ValueError("a non-legacy image setting requires an explicit frame protocol")
        expected_protocol = protocol_id(horizon)
    windows = annotation_window_index(records, horizon)
    candidates = _candidate_actions(action_pool)
    action_to_id = {candidate["text"]: candidate["id"] for candidate in candidates}
    manifest: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("observation index must contain JSON objects")
        if observation.get("protocol") != expected_protocol:
            raise ValueError(
                f"{observation.get('sample_id')} does not use protocol {expected_protocol}"
            )
        if observation.get("split") != split or observation.get("image_setting") != image_setting:
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
            or len(action_list) != horizon
            or not all(isinstance(action, str) for action in action_list)
        ):
            raise ValueError(f"{sample_id} must contain exactly {horizon} action strings")
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
        matched = windows.get(window_key(vid, float(start_f), float(end_f), action_list), [])
        if len(matched) != 1:
            raise ValueError(
                f"{sample_id} matches {len(matched)} annotation windows; expected exactly one"
            )
        source = matched[0]
        manifest_record = {
            "sample_id": sample_id,
            "vid": source["vid"],
            "dataset": source["dataset"],
            "event": source["task_name"],
            "split": split,
            "T": horizon,
            "window_start_step": source["start_step"],
            "window_end_step": source["end_step"],
            "image_setting": image_setting,
            "start_images": _resolved_image_paths(
                observation, frame_root, "start_images", expected_count
            ),
            "end_images": _resolved_image_paths(
                observation, frame_root, "end_images", expected_count
            ),
            "candidate_actions": candidates,
            "gt_action_ids": [action_to_id[action] for action in action_list],
            "gt_actions": action_list,
            "pool_type": "split",
            "candidate_order": "original_file_order",
            "candidate_order_seed": None,
            "candidate_pool_source_size": len(action_pool),
            "candidate_pool_effective_size": len(candidates),
            "candidate_name_normalization": "casefold_whitespace_for_matching",
            "protocol": expected_protocol,
            "response_parser": "legacy_numbered_action_names",
            "source_sequence_index": observation["sequence_index"],
            "source_video_path": observation["source_video_path"],
            "frame_timestamps": observation["frame_timestamps"],
        }
        if "cache_id" in observation:
            manifest_record["frame_cache_id"] = observation["cache_id"]
        if "frame_sha256" in observation:
            manifest_record["frame_sha256"] = observation["frame_sha256"]
        manifest.append(manifest_record)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local, validated manifest from Table V frame observations."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--split", choices=("base", "novel"), required=True)
    parser.add_argument("--horizon", type=int, choices=(3, 4), default=DEFAULT_HORIZON)
    parser.add_argument("--image-setting", choices=("1+1", "3+3"), default="3+3")
    parser.add_argument("--frame-protocol", choices=("legacy", "shared-cache"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.frame_protocol == "legacy" and arguments.image_setting != "3+3":
        parser.error("the legacy Table V protocol only supports --image-setting 3+3")
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    partition = Partition.BASE_TEST if arguments.split == "base" else Partition.NOVEL_TEST
    pool = ActionPool.BASE if arguments.split == "base" else ActionPool.NOVEL
    records = list(bundle.partition_records(partition))
    action_pool = list(bundle.action_pool(pool))
    manifest = build_manifest(
        load_jsonl(arguments.observations),
        records,
        action_pool,
        arguments.frame_root,
        arguments.split,
        arguments.horizon,
        arguments.image_setting,
        (
            protocol_id(arguments.horizon)
            if arguments.frame_protocol == "legacy"
            else TableVFrameProtocol(arguments.horizon, arguments.image_setting).identifier
        ),
    )
    write_jsonl(arguments.output, manifest)
    print(f"Wrote {len(manifest)} records to {arguments.output}")


if __name__ == "__main__":
    main()
