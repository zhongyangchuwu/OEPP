"""Decode each planned Table V frame once into an isolated shared cache."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
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
            if image.format != "JPEG":
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _frame_key(video_path: Path, timestamp: float) -> tuple[str, str]:
    return str(video_path.resolve()), f"{timestamp:.6f}"


def _legacy_image_path(legacy_frame_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("legacy observation frame paths must be non-empty strings")
    root = legacy_frame_root.resolve()
    path = (root / raw_path).resolve()
    if root not in path.parents:
        raise ValueError(f"legacy observation frame path escapes its root: {raw_path}")
    if not _existing_image_is_valid(path):
        raise ValueError(f"legacy observation frame is missing or invalid: {path}")
    return path


def _legacy_frame_sources(
    observations: Iterable[Mapping[str, Any]], legacy_frame_root: Path
) -> dict[tuple[str, str], Path]:
    """Index immutable legacy T=4, 3+3 frames and reject conflicting duplicates."""
    sources: dict[tuple[str, str], Path] = {}
    hashes: dict[tuple[str, str], str] = {}
    for index, observation in enumerate(observations):
        if observation.get("protocol") != "table_v_t4_3x3_legacy_v1":
            raise ValueError(f"legacy observation {index} does not use the T=4 legacy protocol")
        if observation.get("image_setting") != "3+3":
            raise ValueError(f"legacy observation {index} is not a 3+3 observation")
        source_video_path = observation.get("source_video_path")
        timestamps = observation.get("frame_timestamps")
        if not isinstance(source_video_path, str) or not source_video_path:
            raise ValueError(f"legacy observation {index} has no source_video_path")
        if not isinstance(timestamps, Mapping):
            raise ValueError(f"legacy observation {index} has no frame_timestamps mapping")
        for role, image_key in (("start", "start_images"), ("goal", "end_images")):
            role_timestamps = timestamps.get(role)
            image_paths = observation.get(image_key)
            if (
                not isinstance(role_timestamps, list)
                or len(role_timestamps) != 3
                or not all(
                    isinstance(value, (int, float)) and value >= 0 for value in role_timestamps
                )
                or not isinstance(image_paths, list)
                or len(image_paths) != 3
            ):
                raise ValueError(f"legacy observation {index} has an invalid {role} image payload")
            for timestamp, raw_path in zip(role_timestamps, image_paths):
                key = _frame_key(Path(source_video_path), float(timestamp))
                path = _legacy_image_path(legacy_frame_root, raw_path)
                digest = sha256_file(path)
                prior_digest = hashes.get(key)
                if prior_digest is not None and prior_digest != digest:
                    raise ValueError(
                        "legacy observations disagree on JPEG bytes for "
                        f"{key[0]} at {key[1]} seconds"
                    )
                if prior_digest is None or str(path) < str(sources[key]):
                    sources[key] = path
                hashes[key] = digest
    return sources


def _copy_legacy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    try:
        shutil.copyfile(source, temporary)
        if not _existing_image_is_valid(temporary):
            raise ValueError(f"copied legacy frame is invalid: {source}")
        if sha256_file(temporary) != sha256_file(source):
            raise ValueError(f"copied legacy frame hash differs from its source: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _seeded_available_record(
    raw_request: dict[str, Any], destination: Path, source: Path, *, reused: bool
) -> dict[str, Any]:
    record = _available_record(
        raw_request,
        destination,
        seek_timestamp=None,
        decoder_reported_timestamp=None,
        reused=reused,
    )
    record["provenance"] = {
        "kind": "legacy_t4_3x3_frame",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
    }
    return record


def extract_cache_frames(
    requests: Iterable[dict[str, Any]],
    cache_root: Path,
    *,
    legacy_observations: Iterable[Mapping[str, Any]] | None = None,
    legacy_frame_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract planned frames, seeding verified legacy T=4 frames when supplied."""
    if (legacy_observations is None) != (legacy_frame_root is None):
        raise ValueError("legacy observations and legacy frame root must be provided together")
    legacy_sources = (
        _legacy_frame_sources(legacy_observations, legacy_frame_root)
        if legacy_observations is not None and legacy_frame_root is not None
        else {}
    )
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
            _, _, _, request_video_path, request_timestamp, _ = _request_fields(request)
            legacy_source = legacy_sources.get(_frame_key(request_video_path, request_timestamp))
            if legacy_source is not None:
                if _existing_image_is_valid(destination):
                    if sha256_file(destination) != sha256_file(legacy_source):
                        raise ValueError(
                            "existing cached frame differs from its immutable legacy source: "
                            f"{destination}"
                        )
                    outcomes.append(
                        _seeded_available_record(request, destination, legacy_source, reused=True)
                    )
                else:
                    _copy_legacy_image(legacy_source, destination)
                    outcomes.append(
                        _seeded_available_record(request, destination, legacy_source, reused=False)
                    )
                continue
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
    parser.add_argument(
        "--legacy-observations",
        type=Path,
        nargs="+",
        help="immutable legacy T=4, 3+3 observation JSONL files to seed by exact JPEG bytes",
    )
    parser.add_argument(
        "--legacy-frame-root",
        type=Path,
        help="frame root referenced by --legacy-observations",
    )
    arguments = parser.parse_args()
    if (arguments.legacy_observations is None) != (arguments.legacy_frame_root is None):
        parser.error("--legacy-observations and --legacy-frame-root must be supplied together")
    if arguments.frame_index_output.exists():
        raise FileExistsError(
            f"refusing to overwrite frame extraction index: {arguments.frame_index_output}"
        )
    legacy_observations = (
        [observation for path in arguments.legacy_observations for observation in load_jsonl(path)]
        if arguments.legacy_observations is not None
        else None
    )
    outcomes = extract_cache_frames(
        load_jsonl(arguments.frame_requests),
        arguments.cache_root,
        legacy_observations=legacy_observations,
        legacy_frame_root=arguments.legacy_frame_root,
    )
    write_jsonl(arguments.frame_index_output, outcomes)
    available = sum(outcome["status"] == "available" for outcome in outcomes)
    print(f"Extracted or verified {available}/{len(outcomes)} unique planned frames.")


if __name__ == "__main__":
    main()
