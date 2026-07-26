from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import REPOSITORY_ROOT, load_json, load_jsonl, write_json, write_jsonl

SPLIT_FILES = {
    "base": "test_base_dataset_1.json",
    "novel": "novel_dataset_1.json",
}
POOL_FILES = {
    "base": "base_action_pool_1.json",
    "novel": "novel_action_pool_1.json",
    "total": "total_action_pool.json",
}


@dataclass(frozen=True)
class Observation:
    start_images: list[str]
    end_images: list[str]


def _observation_key(
    dataset: str, vid: str, start_step: int, end_step: int, image_setting: str
) -> tuple[str, str, int, int, str]:
    return dataset, vid, start_step, end_step, image_setting


def _expected_image_count(image_setting: str) -> int:
    try:
        start_count, end_count = (int(value) for value in image_setting.split("+"))
    except ValueError as error:
        raise ValueError(f"Invalid image setting: {image_setting}") from error
    if start_count != end_count or start_count not in {1, 3}:
        raise ValueError(f"Only 1+1 and 3+3 are supported, received: {image_setting}")
    return start_count


def _load_observations(path: Path) -> dict[tuple[str, str, int, int, str], Observation]:
    observations: dict[tuple[str, str, int, int, str], Observation] = {}
    for row in load_jsonl(path):
        required = {
            "dataset",
            "vid",
            "start_step",
            "end_step",
            "image_setting",
            "start_images",
            "end_images",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Observation index row misses fields: {', '.join(missing)}")
        image_setting = str(row["image_setting"])
        _expected_image_count(image_setting)
        if not all(isinstance(row[name], str) for name in ("dataset", "vid")):
            raise ValueError("Observation dataset and vid must be strings")
        if type(row["start_step"]) is not int or type(row["end_step"]) is not int:
            raise ValueError("Observation step indices must be integers")
        if not all(isinstance(item, str) for item in row["start_images"] + row["end_images"]):
            raise ValueError("Observation images must be string paths")
        key = _observation_key(
            row["dataset"], row["vid"], row["start_step"], row["end_step"], image_setting
        )
        if key in observations:
            raise ValueError(f"Duplicate observation index row for {key}")
        observations[key] = Observation(row["start_images"], row["end_images"])
    return observations


def _validate_images(observation: Observation, image_setting: str) -> list[str]:
    expected_count = _expected_image_count(image_setting)
    errors: list[str] = []
    if (
        len(observation.start_images) != expected_count
        or len(observation.end_images) != expected_count
    ):
        return [
            f"expected {expected_count} start and {expected_count} end images, got "
            f"{len(observation.start_images)} and {len(observation.end_images)}"
        ]
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies with `uv sync` before validating images."
        ) from error
    for image_path in [*observation.start_images, *observation.end_images]:
        path = Path(image_path)
        if not path.is_file():
            errors.append(f"missing image: {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # Pillow exposes several decoder-specific exception types.
            errors.append(f"unreadable image {path}: {error}")
    return errors


def _candidate_actions(
    action_pool: list[str], candidate_order: str, candidate_order_seed: int | None
) -> list[dict[str, Any]]:
    ordered_actions = list(action_pool)
    if candidate_order == "shuffled":
        if candidate_order_seed is None:
            raise ValueError("A shuffled candidate order requires a seed.")
        random.Random(candidate_order_seed).shuffle(ordered_actions)
    elif candidate_order != "original":
        raise ValueError(f"Unknown candidate order: {candidate_order}")
    if len(ordered_actions) != len(set(ordered_actions)):
        raise ValueError(
            "Candidate action pool contains duplicate strings; cannot create a unique mapping."
        )
    return [{"id": index, "text": action} for index, action in enumerate(ordered_actions)]


def _select_windows(
    records: list[dict[str, Any]], horizon: int, limit: int | None, selection_seed: int
) -> list[tuple[int, int, dict[str, Any]]]:
    windows = [
        (record_index, start_step, record)
        for record_index, record in enumerate(records)
        for start_step in range(max(0, len(record["anno"]) - horizon + 1))
    ]
    if limit is None or limit >= len(windows):
        return windows
    return sorted(
        random.Random(selection_seed).sample(windows, limit), key=lambda item: (item[0], item[1])
    )


def _sample_id(
    split: str, record_index: int, record: dict[str, Any], start_step: int, horizon: int
) -> str:
    return (
        f"{split}_T{horizon}_{record['dataset']}_{record_index:04d}_"
        f"{record['vid']}_{start_step:02d}"
    )


def _candidate_order_tag(candidate_order: str, candidate_order_seed: int | None) -> str:
    return "original" if candidate_order == "original" else f"seed{candidate_order_seed}"


def build_manifest(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    observations = _load_observations(arguments.observation_index)
    pools = {
        name: load_json(arguments.data_root / filename) for name, filename in POOL_FILES.items()
    }
    if not all(
        isinstance(pool, list) and all(isinstance(action, str) for action in pool)
        for pool in pools.values()
    ):
        raise ValueError("Action-pool files must each contain a JSON string list.")

    manifests: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    for split in arguments.split:
        records = load_json(arguments.data_root / SPLIT_FILES[split])
        if not isinstance(records, list):
            raise ValueError(f"{SPLIT_FILES[split]} must contain a JSON list")
        selected_pool = pools["total"] if arguments.pool_type == "total" else pools[split]
        candidates = _candidate_actions(
            selected_pool, arguments.candidate_order, arguments.candidate_order_seed
        )
        action_to_id = {candidate["text"]: candidate["id"] for candidate in candidates}
        candidate_order_tag = _candidate_order_tag(
            arguments.candidate_order, arguments.candidate_order_seed
        )
        for horizon in arguments.horizons:
            windows = _select_windows(
                records, horizon, arguments.limit_per_split, arguments.selection_seed
            )
            for record_index, start_step, record in windows:
                end_step = start_step + horizon - 1
                actions = [step["action"] for step in record["anno"][start_step : end_step + 1]]
                sample_id = _sample_id(split, record_index, record, start_step, horizon)
                for image_setting in arguments.image_settings:
                    observation = observations.get(
                        _observation_key(
                            record["dataset"], record["vid"], start_step, end_step, image_setting
                        )
                    )
                    if observation is None:
                        validation_errors.append(
                            {
                                "sample_id": sample_id,
                                "image_setting": image_setting,
                                "error": "no observation-index row",
                            }
                        )
                        continue
                    image_errors = _validate_images(observation, image_setting)
                    if image_errors:
                        validation_errors.append(
                            {
                                "sample_id": sample_id,
                                "image_setting": image_setting,
                                "error": "; ".join(image_errors),
                            }
                        )
                        continue
                    missing_actions = [action for action in actions if action not in action_to_id]
                    if missing_actions:
                        validation_errors.append(
                            {
                                "sample_id": sample_id,
                                "image_setting": image_setting,
                                "error": (
                                    "ground truth missing from candidate pool: "
                                    f"{missing_actions}"
                                ),
                            }
                        )
                        continue
                    manifests.append(
                        {
                            "sample_id": (
                                f"{sample_id}_{image_setting.replace('+', 'x')}_"
                                f"{arguments.pool_type}_{candidate_order_tag}"
                            ),
                            "vid": record["vid"],
                            "dataset": record["dataset"],
                            "event": record["task_name"],
                            "split": split,
                            "T": horizon,
                            "window_start_step": start_step,
                            "window_end_step": end_step,
                            "image_setting": image_setting,
                            "start_images": observation.start_images,
                            "end_images": observation.end_images,
                            "candidate_actions": candidates,
                            "gt_action_ids": [action_to_id[action] for action in actions],
                            "gt_actions": actions,
                            "pool_type": arguments.pool_type,
                            "candidate_order": (
                                "original_file_order"
                                if arguments.candidate_order == "original"
                                else "seeded_shuffle"
                            ),
                            "candidate_order_seed": arguments.candidate_order_seed,
                        }
                    )
    if validation_errors:
        error_path = arguments.output.with_suffix(".validation_errors.json")
        write_json(error_path, {"errors": validation_errors, "valid_records": len(manifests)})
        raise ValueError(
            f"Manifest not written: {len(validation_errors)} validation error(s). See {error_path}."
        )
    sample_ids = [record["sample_id"] for record in manifests]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Generated manifest has duplicate sample_id values.")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strictly validated OEPP visual API manifests."
    )
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--split", nargs="+", choices=("base", "novel"), required=True)
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 4), required=True)
    parser.add_argument("--image-settings", nargs="+", choices=("1+1", "3+3"), required=True)
    parser.add_argument("--pool-type", choices=("split", "total"), required=True)
    parser.add_argument("--candidate-order", choices=("original", "shuffled"), default="original")
    parser.add_argument("--candidate-order-seed", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.limit_per_split is not None and arguments.limit_per_split <= 0:
        parser.error("--limit-per-split must be positive")
    if arguments.candidate_order == "shuffled" and arguments.candidate_order_seed is None:
        parser.error("--candidate-order-seed is required when --candidate-order shuffled")
    if arguments.candidate_order == "original" and arguments.candidate_order_seed is not None:
        parser.error("--candidate-order-seed is only valid when --candidate-order shuffled")
    try:
        manifest = build_manifest(arguments)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    write_jsonl(arguments.output, manifest)
    print(f"Wrote {len(manifest)} records to {arguments.output}")


if __name__ == "__main__":
    main()
