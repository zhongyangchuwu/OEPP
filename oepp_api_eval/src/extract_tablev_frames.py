from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_json, utc_now, write_jsonl
from PIL import Image

PROTOCOL_ID = "table_v_t4_3x3_legacy_v1"
HORIZON = 4
FRAME_OFFSETS = {"start": (0, 1, 2), "goal": (-2, -1, 0)}


def _sample_id(split: str, index: int, record: dict[str, Any]) -> str:
    return f"tablev_T4_{split}_{index:04d}_{record['vid']}"


def _validate_sequence(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("sequence entry must be a JSON object")
    required = {"vid", "video_path", "start_f", "end_f", "action_list"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"sequence entry is missing fields: {', '.join(missing)}")
    if not isinstance(record["vid"], str) or not isinstance(record["video_path"], str):
        raise ValueError("sequence vid and video_path must be strings")
    if not isinstance(record["start_f"], (int, float)) or not isinstance(
        record["end_f"], (int, float)
    ):
        raise ValueError("sequence frame timestamps must be numeric")
    actions = record["action_list"]
    if (
        not isinstance(actions, list)
        or len(actions) != HORIZON
        or not all(isinstance(action, str) for action in actions)
    ):
        raise ValueError(f"sequence action_list must contain exactly {HORIZON} strings")
    return record


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies with `uv sync` before extracting video frames."
        ) from error
    return cv2


def _resize_to_max_side(image: Image.Image, max_side: int = 512) -> Image.Image:
    width, height = image.size
    scale = max_side / max(width, height)
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def _read_frame(capture: Any, cv2: Any, timestamp: float) -> Image.Image:
    if timestamp < 0:
        raise ValueError(f"frame timestamp must be non-negative, received {timestamp}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        raise ValueError("video reports an invalid frame count or FPS")
    duration = total_frames / fps
    capture.set(cv2.CAP_PROP_POS_MSEC, min(timestamp, duration) * 1000)
    success, frame = capture.read()
    if not success or frame is None:
        raise ValueError(f"cannot decode a frame at {timestamp:.6f} seconds")
    return _resize_to_max_side(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def _write_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        with Image.open(destination) as existing:
            existing.verify()
        return
    temporary = destination.with_suffix(".tmp.jpg")
    try:
        image.save(temporary, format="JPEG")
        with Image.open(temporary) as written:
            written.verify()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_observation(
    record: dict[str, Any], split: str, index: int, frame_root: Path
) -> dict[str, Any]:
    source_video = Path(record["video_path"])
    if not source_video.is_file():
        raise FileNotFoundError(f"source video does not exist: {source_video}")
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {source_video}")
    sample_id = _sample_id(split, index, record)
    output_dir = frame_root / split / sample_id
    paths: dict[str, list[str]] = {"start": [], "goal": []}
    timestamps: dict[str, list[float]] = {"start": [], "goal": []}
    anchors = {"start": float(record["start_f"]), "goal": float(record["end_f"])}
    try:
        for role, offsets in FRAME_OFFSETS.items():
            for position, offset in enumerate(offsets, start=1):
                timestamp = anchors[role] + offset
                image = _read_frame(capture, cv2, timestamp)
                destination = output_dir / f"{role}_{position}.jpg"
                _write_image(image, destination)
                paths[role].append(destination.relative_to(frame_root).as_posix())
                timestamps[role].append(timestamp)
    finally:
        capture.release()
    return {
        "sample_id": sample_id,
        "split": split,
        "sequence_index": index,
        "vid": record["vid"],
        "source_video_path": str(source_video),
        "start_f": float(record["start_f"]),
        "end_f": float(record["end_f"]),
        "action_list": record["action_list"],
        "image_setting": "3+3",
        "protocol": PROTOCOL_ID,
        "frame_offsets": FRAME_OFFSETS,
        "frame_timestamps": timestamps,
        "start_images": paths["start"],
        "end_images": paths["goal"],
        "extracted_at": utc_now(),
    }


def extract(
    sequence_file: Path, split: str, frame_root: Path, limit: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_json(sequence_file)
    if not isinstance(records, list):
        raise ValueError("sequence file must contain a JSON list")
    if limit is not None:
        records = records[:limit]
    observations: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records):
        sample_id = f"tablev_T4_{split}_{index:04d}"
        try:
            record = _validate_sequence(raw_record)
            sample_id = _sample_id(split, index, record)
            observations.append(_extract_observation(record, split, index, frame_root))
        except (OSError, RuntimeError, ValueError) as error:
            unavailable.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "sequence_index": index,
                    "status": "unavailable_observation",
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                }
            )
    return observations, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the exact T=4, 3+3 observations used by the OEPP Table V GPT protocol."
    )
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--split", choices=("base", "novel"), required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path, required=True)
    parser.add_argument("--unavailable-output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    observations, unavailable = extract(
        arguments.sequence_file, arguments.split, arguments.frame_root, arguments.limit
    )
    write_jsonl(arguments.observations_output, observations)
    write_jsonl(arguments.unavailable_output, unavailable)
    print(
        f"Extracted {len(observations)} observation(s); "
        f"recorded {len(unavailable)} unavailable observation(s)."
    )


if __name__ == "__main__":
    main()
