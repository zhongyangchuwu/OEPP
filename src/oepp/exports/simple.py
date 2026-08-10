"""Evaluate deterministic Random and Matching OEPP baselines."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from oepp.baselines.simple import (
    SIMPLE_BASELINE_PROTOCOL,
    matching_action_indices,
    random_action_indices,
)
from oepp.data import (
    FeatureKind,
    FeatureRoots,
    PaddingPolicy,
    Partition,
    SequenceDataset,
    SplitBundle,
)
from oepp.data.features import default_videoclip_root
from oepp.evaluation.planning import summarize_files
from oepp.training.support import action_embedding_tensor, canonical_json_hash, utc_now, write_json

_METHODS = ("random", "matching")
_FIELDS = (
    "sample_id",
    "split_id",
    "split",
    "dataset",
    "task_name",
    "task_id",
    "task_id_old",
    "vid",
    "source_video_index",
    "start_step",
    "end_step",
    "is_padded",
    "pad_count",
    "step_index",
    "gt_action",
    "gt_label",
    "predicted_action",
    "predicted_label",
    "correct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Random and Matching OEPP baselines."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-id", choices=("split-001", "split-002"), required=True)
    parser.add_argument("--horizon", type=int, choices=(3, 4), required=True)
    parser.add_argument("--methods", nargs="+", choices=_METHODS, default=list(_METHODS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _rows_for_predictions(
    dataset: SequenceDataset, predicted_labels: Iterable[Iterable[int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action_pool = list(dataset.action_pool)
    predictions = list(predicted_labels)
    if len(predictions) != len(dataset):
        raise ValueError(f"received {len(predictions)} predictions for {len(dataset)} windows")
    for index, labels in enumerate(predictions):
        metadata = dataset.metadata_at(index)
        predicted = [int(label) for label in labels]
        if len(predicted) != dataset.horizon:
            raise ValueError(f"prediction for {metadata['sample_id']} has wrong horizon")
        if any(label < 0 or label >= len(action_pool) for label in predicted):
            raise ValueError(f"prediction for {metadata['sample_id']} is outside candidate pool")
        ground_actions = list(metadata["actions"])
        ground_labels = [dataset.action_to_label[action] for action in ground_actions]
        for step, (ground_label, predicted_label) in enumerate(zip(ground_labels, predicted)):
            rows.append(
                {
                    **{field: metadata[field] for field in _FIELDS if field in metadata},
                    "step_index": step,
                    "gt_action": action_pool[ground_label],
                    "gt_label": ground_label,
                    "predicted_action": action_pool[predicted_label],
                    "predicted_label": predicted_label,
                    "correct": int(ground_label == predicted_label),
                }
            )
    return rows


def _random_predictions(dataset: SequenceDataset, seed: int) -> list[list[int]]:
    return [
        random_action_indices(
            pool_size=len(dataset.action_pool),
            horizon=dataset.horizon,
            split_id=dataset.bundle.split_id,
            partition=dataset.partition.value,
            sample_id=dataset.metadata_at(index)["sample_id"],
            seed=seed,
        )
        for index in range(len(dataset))
    ]


def _matching_predictions(
    dataset: SequenceDataset, *, batch_size: int, device: torch.device
) -> list[list[int]]:
    candidates = action_embedding_tensor(
        list(dataset.action_pool), dataset.action_embeddings, device
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    output: list[list[int]] = []
    with torch.inference_mode():
        for batch in loader:
            start = batch[1].to(device, non_blocking=True).float()
            goal = batch[2].to(device, non_blocking=True).float()
            dimension = candidates.shape[-1]
            start = start.reshape(start.shape[0], -1, dimension)
            goal = goal.reshape(goal.shape[0], -1, dimension)
            labels = matching_action_indices(start, goal, candidates, horizon=dataset.horizon)
            output.extend(labels.detach().cpu().tolist())
    return output


def _datasets(bundle: SplitBundle, horizon: int) -> dict[str, SequenceDataset]:
    roots = FeatureRoots(videoclip=default_videoclip_root())
    return {
        "base": SequenceDataset(
            bundle,
            Partition.BASE_TEST,
            feature=FeatureKind.VIDEOCLIP,
            horizon=horizon,
            padding=PaddingPolicy.LEFT,
            feature_roots=roots,
        ),
        "novel": SequenceDataset(
            bundle,
            Partition.NOVEL_TEST,
            feature=FeatureKind.VIDEOCLIP,
            horizon=horizon,
            padding=PaddingPolicy.LEFT,
            feature_roots=roots,
        ),
    }


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len(arguments.methods) != len(set(arguments.methods)):
        raise ValueError("--methods must not contain duplicates")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    arguments.output_dir.mkdir(parents=True, exist_ok=False)

    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    datasets = _datasets(bundle, arguments.horizon)
    for method in arguments.methods:
        method_dir = arguments.output_dir / method
        method_dir.mkdir()
        csv_paths: dict[str, Path] = {}
        for split_name, dataset in datasets.items():
            predictions = (
                _random_predictions(dataset, arguments.seed)
                if method == "random"
                else _matching_predictions(dataset, batch_size=arguments.batch_size, device=device)
            )
            rows = _rows_for_predictions(dataset, predictions)
            path = method_dir / f"{split_name}_metrics_per_window_step.csv"
            _write_rows(path, rows)
            csv_paths[split_name] = path
        summary = summarize_files(csv_paths)
        write_json(method_dir / "planning_summary.json", summary)
        write_json(
            method_dir / "run_metadata.json",
            {
                "created_at": utc_now(),
                "schema": "oepp-simple-baseline-run-v1",
                "protocol_id": SIMPLE_BASELINE_PROTOCOL,
                "method": method,
                "definition": (
                    "per-sample SHA-256-derived sampling without replacement"
                    if method == "random"
                    else "endpoint cosine matching with midpoint top-k intermediates"
                ),
                "seed": arguments.seed if method == "random" else None,
                "split_id": bundle.split_id,
                "split_source_hashes": dict(bundle.source_hashes),
                "horizon": arguments.horizon,
                "feature": FeatureKind.VIDEOCLIP.value,
                "padding": PaddingPolicy.LEFT.value,
                "device": str(device),
                "candidate_pool_hashes": {
                    name: canonical_json_hash(list(dataset.action_pool))
                    for name, dataset in datasets.items()
                },
                "windows": {name: len(dataset) for name, dataset in datasets.items()},
                "metrics_csv": {name: str(path) for name, path in csv_paths.items()},
            },
        )
