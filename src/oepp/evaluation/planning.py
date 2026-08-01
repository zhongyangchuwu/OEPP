"""Summarize OEPP planner predictions by both windows and events.

Existing planner metrics are window-micro averages: events with more videos or sliding windows
contribute more. This module retains that result and adds an unweighted event-macro view for
robustness reporting. It consumes the deterministic CSV emitted by both OEPP exporters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

METRIC_SCHEMA = "oepp-planning-event-metrics-v1"
_REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "task_id",
    "step_index",
    "gt_action",
    "predicted_action",
    "correct",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or _REQUIRED_FIELDS.difference(reader.fieldnames):
            missing = sorted(_REQUIRED_FIELDS.difference(reader.fieldnames or ()))
            raise ValueError(f"planning metrics CSV is missing required fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("planning metrics CSV must contain at least one row")
    return rows


def _binary(value: object, label: str) -> int:
    if str(value) not in {"0", "1"}:
        raise ValueError(f"{label} must be encoded as zero or one")
    return int(str(value))


def _window_metrics(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    windows: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("planning metrics row requires a non-empty sample_id")
        windows[sample_id].append(row)

    output: list[dict[str, Any]] = []
    for sample_id, window_rows in sorted(windows.items()):
        event = window_rows[0].get("dataset"), window_rows[0].get("task_id")
        if not all((row.get("dataset"), row.get("task_id")) == event for row in window_rows):
            raise ValueError(f"window {sample_id} has inconsistent event metadata")
        if not all(isinstance(value, str) and value for value in event):
            raise ValueError(f"window {sample_id} requires non-empty dataset and task_id")
        steps: dict[int, Mapping[str, str]] = {}
        for row in window_rows:
            try:
                step = int(str(row.get("step_index")))
            except ValueError as error:
                raise ValueError(f"window {sample_id} has an invalid step_index") from error
            if step < 0 or step in steps:
                raise ValueError(f"window {sample_id} has duplicate or negative step indexes")
            _binary(row.get("correct"), f"window {sample_id} correct")
            if not isinstance(row.get("gt_action"), str) or not isinstance(
                row.get("predicted_action"), str
            ):
                raise ValueError(f"window {sample_id} requires string actions")
            steps[step] = row
        expected_steps = set(range(len(steps)))
        if set(steps) != expected_steps:
            raise ValueError(
                f"window {sample_id} step indexes must start at zero and be contiguous"
            )
        ordered_steps = [steps[index] for index in range(len(steps))]
        truth = {row["gt_action"] for row in ordered_steps}
        predicted = {row["predicted_action"] for row in ordered_steps}
        output.append(
            {
                "sample_id": sample_id,
                "event": (event[0], event[1]),
                "steps": len(ordered_steps),
                "correct_steps": sum(
                    _binary(row["correct"], f"window {sample_id} correct") for row in ordered_steps
                ),
                "sr": float(
                    all(
                        _binary(row["correct"], f"window {sample_id} correct")
                        for row in ordered_steps
                    )
                ),
                "miou": len(truth.intersection(predicted)) / len(truth.union(predicted)),
            }
        )
    return {"windows": output}


def summarize_planning_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    """Return comparable window-micro and event-macro SR, Acc, and mIoU."""
    windows = _window_metrics(rows)["windows"]
    total_windows = len(windows)
    total_steps = sum(window["steps"] for window in windows)
    if total_windows == 0 or total_steps == 0:
        raise ValueError("planning metrics require non-empty windows and steps")
    micro = {
        "windows": total_windows,
        "steps": total_steps,
        "sr": sum(window["sr"] for window in windows) / total_windows,
        "acc": sum(window["correct_steps"] for window in windows) / total_steps,
        "miou": sum(window["miou"] for window in windows) / total_windows,
    }
    by_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_event[window["event"]].append(window)
    events: list[dict[str, Any]] = []
    for (dataset, task_id), event_windows in sorted(by_event.items()):
        event_steps = sum(window["steps"] for window in event_windows)
        events.append(
            {
                "dataset": dataset,
                "task_id": task_id,
                "windows": len(event_windows),
                "steps": event_steps,
                "sr": sum(window["sr"] for window in event_windows) / len(event_windows),
                "acc": sum(window["correct_steps"] for window in event_windows) / event_steps,
                "miou": sum(window["miou"] for window in event_windows) / len(event_windows),
            }
        )
    macro = {
        "events": len(events),
        "sr": sum(event["sr"] for event in events) / len(events),
        "acc": sum(event["acc"] for event in events) / len(events),
        "miou": sum(event["miou"] for event in events) / len(events),
    }
    return {"window_micro": micro, "event_macro": macro, "events": events}


def summarize_files(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != {"base", "novel"}:
        raise ValueError("planning metric summary requires exactly Base and Novel CSV paths")
    return {
        "schema": METRIC_SCHEMA,
        "splits": {
            name: {
                "metrics_csv": str(path),
                "metrics_csv_sha256": _sha256(path),
                **summarize_planning_rows(_read_rows(path)),
            }
            for name, path in sorted(paths.items())
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report separate window-micro and event-macro OEPP planning metrics."
    )
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--novel-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite planning metric summary: {arguments.output}")
    report = summarize_files({"base": arguments.base_csv, "novel": arguments.novel_csv})
    _write_json(arguments.output, report)
    print(
        json.dumps(
            {
                name: {
                    "window_micro": values["window_micro"],
                    "event_macro": values["event_macro"],
                }
                for name, values in report["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
