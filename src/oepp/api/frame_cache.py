"""Versioned, shared frame-cache planning for Table V visual inputs.

The cache is derived from immutable OEPP annotations and a separate video-source index.
It never alters historical frame assets, manifests, or evaluation runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oepp.data import Partition, SplitBundle

from .common import load_json, sha256_file, utc_now, write_json, write_jsonl
from .tablev import TableVFrameProtocol, annotation_window_index, annotation_windows, window_key

FRAME_CACHE_SCHEMA = "oepp-table-v-shared-frame-cache-v2"
VIDEO_SOURCE_INDEX_SCHEMA = "oepp-table-v-video-source-index-v1"
FRAME_TRANSFORM = {
    "format": "JPEG",
    "max_side": 512,
    "resampling": "lanczos",
}


@dataclass(frozen=True)
class VideoSource:
    dataset: str
    vid: str
    video_path: str

    @property
    def identity(self) -> tuple[str, str]:
        return self.dataset, self.vid


@dataclass(frozen=True)
class FrameRequest:
    frame_id: str
    dataset: str
    vid: str
    video_path: str
    requested_timestamp: float
    relative_path: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_timestamp(value: float) -> str:
    return f"{value:.6f}"


def _frame_id(dataset: str, vid: str, timestamp: float) -> str:
    identity = {
        "dataset": dataset,
        "vid": vid,
        "requested_timestamp": _canonical_timestamp(timestamp),
        "transform": FRAME_TRANSFORM,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _partition_for_split(split: str) -> Partition:
    if split == "base":
        return Partition.BASE_TEST
    if split == "novel":
        return Partition.NOVEL_TEST
    raise ValueError(f"unsupported Table V split: {split!r}")


def _require_video_source(value: Mapping[str, Any]) -> VideoSource:
    dataset = value.get("dataset")
    vid = value.get("vid")
    video_path = value.get("video_path")
    if not all(isinstance(item, str) and item for item in (dataset, vid, video_path)):
        raise ValueError(
            "video source entries require non-empty dataset, vid, and video_path strings"
        )
    return VideoSource(dataset, vid, video_path)


def load_video_source_index(path: Path, split_id: str) -> dict[tuple[str, str], VideoSource]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("video-source index must be a JSON object")
    if payload.get("schema") != VIDEO_SOURCE_INDEX_SCHEMA:
        raise ValueError(f"video-source index must use {VIDEO_SOURCE_INDEX_SCHEMA}")
    if payload.get("split_id") != split_id:
        raise ValueError("video-source index split_id does not match the selected split bundle")
    source_rows = payload.get("sources")
    if not isinstance(source_rows, list):
        raise ValueError("video-source index sources must be a JSON list")
    sources: dict[tuple[str, str], VideoSource] = {}
    for raw_source in source_rows:
        if not isinstance(raw_source, Mapping):
            raise ValueError("video-source index contains a non-object source")
        source = _require_video_source(raw_source)
        if source.identity in sources:
            raise ValueError(f"video-source index duplicates {source.identity}")
        sources[source.identity] = source
    return sources


def build_video_source_index(
    bundle: SplitBundle, sequence_files: Mapping[str, Path]
) -> dict[str, Any]:
    """Resolve immutable `(dataset, vid) → video_path` entries from legacy sequence files."""
    if set(sequence_files) != {"base", "novel"}:
        raise ValueError("video-source index requires exactly Base and Novel sequence files")
    source_by_identity: dict[tuple[str, str], VideoSource] = {}
    for split, sequence_file in sequence_files.items():
        rows = load_json(sequence_file)
        if not isinstance(rows, list):
            raise ValueError(f"{sequence_file} must contain a JSON list")
        partition = _partition_for_split(split)
        records = bundle.partition_records(partition)
        window_indexes = {horizon: annotation_window_index(records, horizon) for horizon in (3, 4)}
        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"{sequence_file} row {index} is not an object")
            vid = raw_row.get("vid")
            start = raw_row.get("start_f")
            end = raw_row.get("end_f")
            actions = raw_row.get("action_list")
            video_path = raw_row.get("video_path")
            if (
                not isinstance(vid, str)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not isinstance(actions, list)
                or len(actions) not in {3, 4}
                or not all(isinstance(action, str) for action in actions)
                or not isinstance(video_path, str)
                or not video_path
            ):
                raise ValueError(f"{sequence_file} row {index} has invalid Table V identity")
            identity = window_key(vid, float(start), float(end), actions)
            matches = window_indexes[len(actions)].get(identity, [])
            if len(matches) != 1:
                raise ValueError(
                    f"{sequence_file} row {index} matches {len(matches)} frozen {split} windows; "
                    "expected exactly one"
                )
            record = matches[0]
            source = VideoSource(record["dataset"], vid, video_path)
            prior = source_by_identity.get(source.identity)
            if prior is not None and prior.video_path != source.video_path:
                raise ValueError(f"conflicting video paths for {source.identity}")
            source_by_identity[source.identity] = source
    expected = {
        (record["dataset"], record["vid"])
        for split in ("base", "novel")
        for record in bundle.partition_records(_partition_for_split(split))
    }
    missing = sorted(expected.difference(source_by_identity))
    if missing:
        raise ValueError(f"video-source index lacks {len(missing)} frozen videos: {missing[:3]}")
    return {
        "schema": VIDEO_SOURCE_INDEX_SCHEMA,
        "generated_at": utc_now(),
        "split_id": bundle.split_id,
        "source_hashes": dict(bundle.source_hashes),
        "sequence_files": {split: str(path) for split, path in sorted(sequence_files.items())},
        "sources": [asdict(source) for _, source in sorted(source_by_identity.items())],
    }


def _observation_id(
    protocol: TableVFrameProtocol,
    split: str,
    record_index: int,
    record: Mapping[str, Any],
    start_step: int,
) -> str:
    return (
        f"tablev_cache_{protocol.identifier}_{split}_{record_index:04d}_"
        f"{record['vid']}_{start_step:02d}"
    )


def build_cache_plan(
    bundle: SplitBundle,
    sources: Mapping[tuple[str, str], VideoSource],
    protocols: Sequence[TableVFrameProtocol],
    cache_id: str,
) -> tuple[list[FrameRequest], list[dict[str, Any]], dict[str, Any]]:
    """Plan each required timestamp once and compose all protocol observations by reference."""
    if not cache_id or Path(cache_id).name != cache_id:
        raise ValueError("cache_id must be a non-empty path component")
    if not protocols:
        raise ValueError("at least one Table V frame protocol is required")
    requests: dict[str, FrameRequest] = {}
    observations: list[dict[str, Any]] = []
    for split in ("base", "novel"):
        records = bundle.partition_records(_partition_for_split(split))
        for record_index, record in enumerate(records):
            identity = (record["dataset"], record["vid"])
            try:
                source = sources[identity]
            except KeyError as error:
                raise ValueError(f"video-source index has no entry for {identity}") from error
            for protocol in protocols:
                for window in annotation_windows(record, protocol.horizon):
                    references: dict[str, list[str]] = {"start": [], "goal": []}
                    timestamps: dict[str, list[float]] = {"start": [], "goal": []}
                    for role, offsets in protocol.offsets.items():
                        anchor = window["start_f"] if role == "start" else window["end_f"]
                        for offset in offsets:
                            timestamp = float(anchor) + offset
                            frame_id = _frame_id(source.dataset, source.vid, timestamp)
                            request = requests.setdefault(
                                frame_id,
                                FrameRequest(
                                    frame_id=frame_id,
                                    dataset=source.dataset,
                                    vid=source.vid,
                                    video_path=source.video_path,
                                    requested_timestamp=timestamp,
                                    relative_path=f"frames/{frame_id}.jpg",
                                ),
                            )
                            references[role].append(request.frame_id)
                            timestamps[role].append(timestamp)
                    observations.append(
                        {
                            "schema": FRAME_CACHE_SCHEMA,
                            "cache_id": cache_id,
                            "protocol": protocol.identifier,
                            "image_setting": protocol.image_setting,
                            "sample_id": _observation_id(
                                protocol,
                                split,
                                record_index,
                                record,
                                window["start_step"],
                            ),
                            "split": split,
                            "dataset": source.dataset,
                            "vid": source.vid,
                            "event": record["task_name"],
                            "source_video_path": source.video_path,
                            "source_record_index": record_index,
                            "start_step": window["start_step"],
                            "end_step": window["end_step"],
                            "start_f": window["start_f"],
                            "end_f": window["end_f"],
                            "action_list": window["action_list"],
                            "frame_ids": references,
                            "frame_timestamps": timestamps,
                        }
                    )
    metadata = {
        "schema": FRAME_CACHE_SCHEMA,
        "cache_id": cache_id,
        "generated_at": utc_now(),
        "split_id": bundle.split_id,
        "source_hashes": dict(bundle.source_hashes),
        "frame_transform": FRAME_TRANSFORM,
        "protocols": [protocol.identifier for protocol in protocols],
        "frame_request_count": len(requests),
        "observation_count": len(observations),
    }
    return [request for _, request in sorted(requests.items())], observations, metadata


def write_cache_plan(
    output_dir: Path,
    requests: Iterable[FrameRequest],
    observations: Iterable[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"cache-plan output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "frame_requests.jsonl", [request.as_json() for request in requests])
    write_jsonl(output_dir / "observations.jsonl", observations)
    write_json(output_dir / "metadata.json", dict(metadata))


def build_video_index_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a read-only Table V video-source index from frozen sequence JSON files."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument("--base-sequences", type=Path, required=True)
    parser.add_argument("--novel-sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    index = build_video_source_index(
        bundle,
        {"base": arguments.base_sequences, "novel": arguments.novel_sequences},
    )
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite video-source index: {arguments.output}")
    write_json(arguments.output, index)
    print(f"Wrote {len(index['sources'])} video sources to {arguments.output}")


def plan_cache_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plan a versioned shared Table V frame cache without reading videos."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--horizons", type=int, choices=(3, 4), nargs="+", required=True)
    parser.add_argument("--image-settings", choices=("1+1", "3+3"), nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    sources = load_video_source_index(arguments.video_index, bundle.split_id)
    protocols = [
        TableVFrameProtocol(horizon, image_setting)
        for horizon in arguments.horizons
        for image_setting in arguments.image_settings
    ]
    requests, observations, metadata = build_cache_plan(
        bundle, sources, protocols, arguments.cache_id
    )
    metadata = {
        **metadata,
        "video_index_path": str(arguments.video_index),
        "video_index_sha256": sha256_file(arguments.video_index),
    }
    write_cache_plan(arguments.output_dir, requests, observations, metadata)
    print(
        f"Planned {metadata['frame_request_count']} unique frames and "
        f"{metadata['observation_count']} observations under {arguments.output_dir}."
    )
