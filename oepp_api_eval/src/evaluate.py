from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_jsonl, utc_now, write_json, write_jsonl


def _metric(predicted: list[str] | None, truth: list[str]) -> dict[str, Any]:
    if predicted is None or len(predicted) != len(truth):
        return {"sr": False, "acc": 0.0, "iou": 0.0}
    correct = sum(prediction == target for prediction, target in zip(predicted, truth, strict=True))
    predicted_set, truth_set = set(predicted), set(truth)
    return {
        "sr": predicted == truth,
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


def evaluate(
    manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        action_ids = prediction.get("action_ids") if prediction else None
        parsed_actions = (
            [candidate_text[action_id] for action_id in action_ids]
            if prediction
            and prediction.get("parse_status") == "ok"
            and isinstance(action_ids, list)
            else None
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
    parse_failures = sum(row["prediction_status"] != "ok" for row in rows)
    report = {
        "generated_at": utc_now(),
        "overall": _aggregate(rows),
        "by_split": {
            split: _aggregate(split_rows) for split, split_rows in sorted(by_split.items())
        },
        "by_event": event_metrics,
        "event_macro": macro,
        "parse_or_missing_failure_rate": 100.0 * parse_failures / len(rows) if rows else 0.0,
        "parse_or_missing_failures": parse_failures,
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
    arguments = parser.parse_args()
    rows, report = evaluate(load_jsonl(arguments.manifest), load_jsonl(arguments.predictions))
    write_json(arguments.output, report)
    if arguments.per_sample_output:
        write_jsonl(arguments.per_sample_output, rows)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
