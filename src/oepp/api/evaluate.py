from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import load_jsonl, utc_now, write_json, write_jsonl

STRICT_SCORING = "strict"
PAPER_COMPATIBLE_SCORING = "paper_compatible"
SCORING_MODES = frozenset({STRICT_SCORING, PAPER_COMPATIBLE_SCORING})


def _action_key(action: str) -> str:
    return "".join(action.casefold().split())


def _metric(predicted: list[str] | None, truth: list[str]) -> dict[str, Any]:
    if predicted is None or len(predicted) != len(truth):
        return {"sr": False, "acc": 0.0, "iou": 0.0}
    normalized_predicted = [_action_key(action) for action in predicted]
    normalized_truth = [_action_key(action) for action in truth]
    correct = sum(
        prediction == target
        for prediction, target in zip(normalized_predicted, normalized_truth, strict=True)
    )
    predicted_set, truth_set = set(normalized_predicted), set(normalized_truth)
    return {
        "sr": normalized_predicted == normalized_truth,
        "acc": correct / len(truth),
        "iou": 100.0 * len(predicted_set & truth_set) / len(predicted_set | truth_set),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    if not total:
        return {"samples": 0, "SR": 0.0, "Acc": 0.0, "mIoU": 0.0}
    return {
        "samples": total,
        "SR": 100.0 * sum(record["sr"] for record in records) / total,
        "Acc": 100.0 * sum(record["acc"] for record in records) / total,
        "mIoU": sum(record["iou"] for record in records) / total,
    }


def _action_texts(prediction: dict[str, Any]) -> list[str] | None:
    action_texts = prediction.get("action_texts")
    if isinstance(action_texts, list) and all(isinstance(action, str) for action in action_texts):
        return action_texts
    return None


def _prediction_actions(
    prediction: dict[str, Any] | None,
    candidate_text: dict[int, str],
    scoring_mode: str,
) -> tuple[list[str] | None, bool]:
    if prediction is None:
        return None, False
    action_texts = _action_texts(prediction)
    if scoring_mode == PAPER_COMPATIBLE_SCORING and action_texts is not None:
        return action_texts, True
    action_ids = prediction.get("action_ids")
    if prediction.get("parse_status") != "ok" or not isinstance(action_ids, list):
        return None, False
    if not all(type(action_id) is int and action_id in candidate_text for action_id in action_ids):
        return None, False
    return [candidate_text[action_id] for action_id in action_ids], False


def evaluate(
    manifest: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scoring_mode: str = STRICT_SCORING,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"unsupported scoring mode: {scoring_mode}")
    predictions_by_sample = {record["sample_id"]: record for record in predictions}
    if len(predictions_by_sample) != len(predictions):
        raise ValueError("Predictions JSONL contains duplicate sample_id values.")
    rows: list[dict[str, Any]] = []
    for sample in manifest:
        candidates = sample.get("candidate_actions")
        if not isinstance(candidates, list):
            raise ValueError(f"{sample['sample_id']} lacks candidate_actions")
        candidate_text = {candidate["id"]: candidate["text"] for candidate in candidates}
        prediction = predictions_by_sample.get(sample["sample_id"])
        parsed_actions, scored_from_raw_action_texts = _prediction_actions(
            prediction, candidate_text, scoring_mode
        )
        metrics = _metric(parsed_actions, sample["gt_actions"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "vid": sample["vid"],
                "event": sample["event"],
                "split": sample["split"],
                "T": sample["T"],
                "image_setting": sample["image_setting"],
                "pool_type": sample["pool_type"],
                "candidate_order_seed": sample["candidate_order_seed"],
                "list_pred": parsed_actions,
                "list_true": sample["gt_actions"],
                "prediction_status": prediction.get("parse_status")
                if prediction
                else "missing_prediction",
                "scored_from_raw_action_texts": scored_from_raw_action_texts,
                **metrics,
            }
        )
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[row["event"]].append(row)
        by_split[row["split"]].append(row)
    event_metrics = {
        event: _aggregate(event_rows) for event, event_rows in sorted(by_event.items())
    }
    macro = {
        key: sum(metrics[key] for metrics in event_metrics.values()) / len(event_metrics)
        if event_metrics
        else 0.0
        for key in ("SR", "Acc", "mIoU")
    }
    strict_failures = sum(row["prediction_status"] != "ok" for row in rows)
    raw_scored = sum(row["scored_from_raw_action_texts"] for row in rows)
    report = {
        "generated_at": utc_now(),
        "scoring_mode": scoring_mode,
        "overall": _aggregate(rows),
        "by_split": {
            split: _aggregate(split_rows) for split, split_rows in sorted(by_split.items())
        },
        "by_event": event_metrics,
        "event_macro": macro,
        "parse_or_missing_failure_rate": 100.0 * strict_failures / len(rows) if rows else 0.0,
        "parse_or_missing_failures": strict_failures,
        "paper_compatible_raw_action_sequences": raw_scored,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OEPP API predictions without excluding failures."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--scoring-mode", choices=sorted(SCORING_MODES), default=STRICT_SCORING)
    arguments = parser.parse_args()
    rows, report = evaluate(
        load_jsonl(arguments.manifest), load_jsonl(arguments.predictions), arguments.scoring_mode
    )
    write_json(arguments.output, report)
    if arguments.per_sample_output:
        write_jsonl(arguments.per_sample_output, rows)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
