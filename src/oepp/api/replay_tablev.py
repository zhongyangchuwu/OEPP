"""Replay legacy Table V action lists against frozen OEPP annotations without API calls."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oepp.data import ActionPool, Partition, SplitBundle

from .common import load_json, utc_now, write_json, write_jsonl
from .evaluate import PAPER_COMPATIBLE_SCORING, STRICT_SCORING, evaluate
from .tablev import action_key, annotation_window_index, require_horizon, window_key

REPLAY_SCHEMA = "oepp-table-v-historical-replay-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_actions(value: Any, field: str, horizon: int, *, required: bool) -> list[str] | None:
    if (
        not isinstance(value, list)
        or len(value) != horizon
        or not all(isinstance(action, str) for action in value)
    ):
        if required:
            raise ValueError(f"legacy result {field} must contain exactly {horizon} strings")
        return None
    return value


def _candidate_maps(
    pool: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[int]]]:
    candidates = [{"id": index, "text": action} for index, action in enumerate(pool)]
    exact_ids = {action: index for index, action in enumerate(pool)}
    normalized_ids: dict[str, list[int]] = {}
    for candidate in candidates:
        normalized_ids.setdefault(action_key(candidate["text"]), []).append(candidate["id"])
    return candidates, exact_ids, normalized_ids


def _strict_prediction(
    actions: list[str] | None, candidate_ids: Mapping[str, list[int]]
) -> tuple[list[int] | None, str]:
    if actions is None:
        return None, "legacy_invalid_output"
    identifiers: list[int] = []
    for action in actions:
        matches = candidate_ids.get(action_key(action), [])
        if len(matches) != 1:
            return None, "legacy_out_of_pool_or_ambiguous_output"
        identifiers.append(matches[0])
    return identifiers, "ok"


def _partition(split: str) -> tuple[Partition, ActionPool]:
    if split == "base":
        return Partition.BASE_TEST, ActionPool.BASE
    if split == "novel":
        return Partition.NOVEL_TEST, ActionPool.NOVEL
    raise ValueError(f"unsupported Table V split: {split!r}")


def build_replay_inputs(
    legacy_results: Sequence[Any], bundle: SplitBundle, split: str, horizon: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Match legacy rows to frozen source windows and prepare dual-score evaluator inputs."""
    horizon = require_horizon(horizon)
    partition, pool_type = _partition(split)
    records = bundle.partition_records(partition)
    windows = annotation_window_index(records, horizon)
    pool = bundle.action_pool(pool_type)
    candidates, exact_ids, normalized_ids = _candidate_maps(pool)
    manifest: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    seen_windows: set[tuple[str, float, float, tuple[str, ...]]] = set()

    for index, raw_result in enumerate(legacy_results):
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"legacy result at index {index} is not an object")
        vid = raw_result.get("vid")
        start = raw_result.get("start_f")
        end = raw_result.get("end_f")
        if (
            not isinstance(vid, str)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"legacy result at index {index} has invalid identity")
        truth = _result_actions(
            raw_result.get("action_list"), "action_list", horizon, required=True
        )
        assert truth is not None
        if any(action not in exact_ids for action in truth):
            raise ValueError(
                f"legacy result at index {index} has ground truth outside the {split} pool"
            )
        identity = window_key(vid, float(start), float(end), truth)
        matches = windows.get(identity, [])
        if len(matches) != 1:
            raise ValueError(
                f"legacy result at index {index} matches {len(matches)} frozen {split} windows; "
                "expected exactly one"
            )
        if identity in seen_windows:
            raise ValueError(f"legacy result at index {index} duplicates a source window")
        seen_windows.add(identity)
        source = matches[0]
        sample_id = f"historical_tablev_T{horizon}_{split}_{index:04d}_{vid}"
        output = _result_actions(
            raw_result.get("output_list"), "output_list", horizon, required=False
        )
        action_ids, parse_status = _strict_prediction(output, normalized_ids)
        manifest.append(
            {
                "sample_id": sample_id,
                "vid": source["vid"],
                "event": source["task_name"],
                "split": split,
                "T": horizon,
                "image_setting": "3+3",
                "pool_type": "split",
                "candidate_order_seed": None,
                "candidate_actions": candidates,
                "gt_actions": truth,
            }
        )
        predictions.append(
            {
                "sample_id": sample_id,
                "parse_status": parse_status,
                "action_ids": action_ids,
                "action_texts": output,
            }
        )

    coverage = {
        "source_window_count": sum(len(matches) for matches in windows.values()),
        "legacy_result_count": len(legacy_results),
        "matched_window_count": len(seen_windows),
        "missing_source_window_count": sum(len(matches) for matches in windows.values())
        - len(seen_windows),
    }
    return manifest, predictions, coverage


def replay(
    results_path: Path, bundle: SplitBundle, split: str, horizon: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_results = load_json(results_path)
    if not isinstance(raw_results, list):
        raise ValueError("legacy results file must contain a JSON list")
    manifest, predictions, coverage = build_replay_inputs(raw_results, bundle, split, horizon)
    strict_rows, strict_metrics = evaluate(manifest, predictions, STRICT_SCORING)
    paper_rows, paper_metrics = evaluate(manifest, predictions, PAPER_COMPATIBLE_SCORING)
    paper_by_id = {row["sample_id"]: row for row in paper_rows}
    rows = [
        {
            "sample_id": strict_row["sample_id"],
            "vid": strict_row["vid"],
            "event": strict_row["event"],
            "split": strict_row["split"],
            "T": strict_row["T"],
            "list_true": strict_row["list_true"],
            "strict": {
                key: strict_row[key]
                for key in ("list_pred", "prediction_status", "sr", "acc", "iou")
            },
            "paper_compatible": {
                key: paper_by_id[strict_row["sample_id"]][key]
                for key in ("list_pred", "prediction_status", "sr", "acc", "iou")
            },
        }
        for strict_row in strict_rows
    ]
    report = {
        "schema": REPLAY_SCHEMA,
        "generated_at": utc_now(),
        "legacy_results_path": str(results_path),
        "legacy_results_sha256": _sha256(results_path),
        "split_id": bundle.split_id,
        "split": split,
        "horizon": horizon,
        "source_hashes": dict(bundle.source_hashes),
        "coverage": coverage,
        "strict": strict_metrics,
        "paper_compatible": paper_metrics,
    }
    return rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay legacy Table V action lists against frozen OEPP windows without API calls."
        )
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument("--split", choices=("base", "novel"), required=True)
    parser.add_argument("--horizon", type=int, choices=(3, 4), required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    rows, report = replay(arguments.results, bundle, arguments.split, arguments.horizon)
    write_jsonl(arguments.rows_output, rows)
    write_json(arguments.metrics_output, report)
    print(
        f"Replayed {report['coverage']['matched_window_count']} / "
        f"{report['coverage']['source_window_count']} frozen {arguments.split} windows."
    )


if __name__ == "__main__":
    main()
