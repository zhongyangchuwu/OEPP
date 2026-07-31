"""Decode each planned Table V frame once into an isolated shared cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from .common import load_jsonl, sha256_file, utc_now, write_jsonl
from .extract_tablev_frames import _cv2, _read_frame_with_metadata, _write_image
from .frame_cache import FRAME_CACHE_SCHEMA


def _request_fields(raw_request: dict[str, Any]) -> tuple[str, str, str, Path, float, str]:
    frame_id = raw_request.get("frame_id")
    dataset = raw_request.get("dataset")
    vid = raw_request.get("vid")
    video_path = raw_request.get("video_path")
    timestamp = raw_request.get("requested_timestamp")
    relative_path = raw_request.get("relative_path")
    required_strings = (frame_id, dataset, vid, video_path, relative_path)
    if not all(isinstance(item, str) and item for item in required_strings):
        raise ValueError(
            "frame request requires non-empty frame_id, dataset, vid, video_path, and relative_path"
        )
    if not isinstance(timestamp, (int, float)) or timestamp < 0:
        raise ValueError(f"frame request {frame_id} has an invalid requested_timestamp")
    return frame_id, dataset, vid, Path(video_path), float(timestamp), relative_path


def _destination(cache_root: Path, relative_path: str) -> Path:
    root = cache_root.resolve()
    destination = (root / relative_path).resolve()
    if root not in destination.parents:
        raise ValueError(f"frame cache path escapes --cache-root: {relative_path}")
    return destination


def _available_record(
    raw_request: dict[str, Any],
    destination: Path,
    *,
    seek_timestamp: float | None,
    decoder_reported_timestamp: float | None,
    reused: bool,
) -> dict[str, Any]:
    frame_id, dataset, vid, video_path, timestamp, relative_path = _request_fields(raw_request)
    return {
        "schema": FRAME_CACHE_SCHEMA,
        "frame_id": frame_id,
        "dataset": dataset,
        "vid": vid,
        "video_path": str(video_path),
        "relative_path": relative_path,
        "requested_timestamp": timestamp,
        "seek_timestamp": seek_timestamp,
        "decoder_reported_timestamp": decoder_reported_timestamp,
        "status": "available",
        "reused": reused,
        "image_sha256": sha256_file(destination),
        "extracted_at": utc_now(),
    }


def _unavailable_record(raw_request: dict[str, Any], error: Exception) -> dict[str, Any]:
    frame_id, dataset, vid, video_path, timestamp, relative_path = _request_fields(raw_request)
    return {
        "schema": FRAME_CACHE_SCHEMA,
        "frame_id": frame_id,
        "dataset": dataset,
        "vid": vid,
        "video_path": str(video_path),
        "relative_path": relative_path,
        "requested_timestamp": timestamp,
        "status": "unavailable_frame",
        "error_type": error.__class__.__name__,
        "error": str(error),
        "extracted_at": utc_now(),
    }


def _existing_image_is_valid(destination: Path) -> bool:
    if not destination.is_file():
        return False
    try:
        with Image.open(destination) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def extract_cache_frames(
    requests: Iterable[dict[str, Any]], cache_root: Path
) -> list[dict[str, Any]]:
    """Extract planned frames in source-video batches, retaining every request outcome."""
    normalized_requests = list(requests)
    seen_ids: set[str] = set()
    by_video: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for raw_request in normalized_requests:
        frame_id, _, _, video_path, _, relative_path = _request_fields(raw_request)
        if frame_id in seen_ids:
            raise ValueError(f"frame plan duplicates frame_id {frame_id}")
        seen_ids.add(frame_id)
        _destination(cache_root, relative_path)
        by_video[video_path].append(raw_request)

    outcomes: list[dict[str, Any]] = []
    cv2: Any | None = None
    for video_path in sorted(by_video, key=str):
        video_requests = sorted(by_video[video_path], key=lambda request: request["frame_id"])
        pending: list[dict[str, Any]] = []
        for request in video_requests:
            destination = _destination(cache_root, request["relative_path"])
            if _existing_image_is_valid(destination):
                outcomes.append(
                    _available_record(
                        request,
                        destination,
                        seek_timestamp=None,
                        decoder_reported_timestamp=None,
                        reused=True,
                    )
                )
            else:
                pending.append(request)
        if not pending:
            continue
        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"source video does not exist: {video_path}")
            if cv2 is None:
                cv2 = _cv2()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"cannot open source video: {video_path}")
        except (OSError, RuntimeError, ValueError) as error:
            outcomes.extend(_unavailable_record(request, error) for request in pending)
            continue
        try:
            for request in pending:
                destination = _destination(cache_root, request["relative_path"])
                try:
                    image, seek_timestamp, reported_timestamp = _read_frame_with_metadata(
                        capture, cv2, request["requested_timestamp"]
                    )
                    _write_image(image, destination)
                    outcomes.append(
                        _available_record(
                            request,
                            destination,
                            seek_timestamp=seek_timestamp,
                            decoder_reported_timestamp=reported_timestamp,
                            reused=False,
                        )
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    outcomes.append(_unavailable_record(request, error))
        finally:
            capture.release()
    return sorted(outcomes, key=lambda outcome: outcome["frame_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a planned Table V shared frame cache without changing legacy assets."
    )
    parser.add_argument("--frame-requests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--frame-index-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.frame_index_output.exists():
        raise FileExistsError(
            f"refusing to overwrite frame extraction index: {arguments.frame_index_output}"
        )
    outcomes = extract_cache_frames(load_jsonl(arguments.frame_requests), arguments.cache_root)
    write_jsonl(arguments.frame_index_output, outcomes)
    available = sum(outcome["status"] == "available" for outcome in outcomes)
    print(f"Extracted or verified {available}/{len(outcomes)} unique planned frames.")


if __name__ == "__main__":
    main()
