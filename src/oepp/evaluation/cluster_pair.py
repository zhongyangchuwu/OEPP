"""Aggregate paired same-cluster-supported and cluster-blocked OEPP predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oepp.data.cluster_pair import PAIR_CONDITIONS, PAIR_PLAN_SCHEMA

from .planning import summarize_planning_rows

SUMMARY_SCHEMA = "oepp-q32-cluster-pair-metrics-v1"
_REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "task_name",
    "task_id",
    "vid",
    "start_step",
    "end_step",
    "step_index",
    "gt_action",
    "predicted_action",
    "correct",
}
_METRICS = ("sr", "acc", "miou")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or _REQUIRED_FIELDS.difference(reader.fieldnames):
            missing = sorted(_REQUIRED_FIELDS.difference(reader.fieldnames or ()))
            raise ValueError(f"cluster-pair CSV is missing fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"cluster-pair CSV is empty: {path}")
    return rows


def _pair_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        str(row[field])
        for field in (
            "dataset",
            "task_id",
            "vid",
            "start_step",
            "end_step",
            "step_index",
        )
    )


def _validate_pair(
    supported: list[dict[str, str]],
    blocked: list[dict[str, str]],
    target_events: set[str],
) -> None:
    for name, rows in (("supported", supported), ("blocked", blocked)):
        observed = {row["task_name"] for row in rows}
        if observed != target_events:
            raise ValueError(
                f"{name} rows do not match target events: expected {sorted(target_events)}, "
                f"got {sorted(observed)}"
            )
    supported_by_key = {_pair_key(row): row for row in supported}
    blocked_by_key = {_pair_key(row): row for row in blocked}
    if len(supported_by_key) != len(supported) or len(blocked_by_key) != len(blocked):
        raise ValueError("paired rows contain duplicate evaluation keys")
    if set(supported_by_key) != set(blocked_by_key):
        raise ValueError("paired conditions do not contain identical target windows and steps")
    for key, row in supported_by_key.items():
        other = blocked_by_key[key]
        if row["gt_action"] != other["gt_action"] or row["task_name"] != other["task_name"]:
            raise ValueError(f"paired ground truth differs for {key}")


def _event_stats(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    summary = summarize_planning_rows(rows)
    return {
        (str(event["dataset"]), str(event["task_id"])): {
            "windows": float(event["windows"]),
            "steps": float(event["steps"]),
            "sr": float(event["sr"]),
            "acc": float(event["acc"]),
            "miou": float(event["miou"]),
        }
        for event in summary["events"]
    }


def _aggregate_sample(
    stats: Mapping[tuple[str, str], Mapping[str, float]],
    sampled_events: np.ndarray,
) -> dict[str, dict[str, float]]:
    selected = [stats[tuple(event)] for event in sampled_events.tolist()]
    windows = sum(event["windows"] for event in selected)
    steps = sum(event["steps"] for event in selected)
    return {
        "window_micro": {
            "sr": sum(event["sr"] * event["windows"] for event in selected) / windows,
            "acc": sum(event["acc"] * event["steps"] for event in selected) / steps,
            "miou": sum(event["miou"] * event["windows"] for event in selected) / windows,
        },
        "event_macro": {
            metric: sum(event[metric] for event in selected) / len(selected)
            for metric in _METRICS
        },
    }


def _paired_bootstrap(
    supported: Mapping[tuple[str, str], Mapping[str, float]],
    blocked: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    seed: int,
    repeats: int,
) -> dict[str, dict[str, dict[str, float]]]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    events = tuple(sorted(supported))
    if set(events) != set(blocked):
        raise ValueError("paired bootstrap requires identical event keys")
    rng = np.random.default_rng(seed)
    estimates = {
        aggregation: {metric: [] for metric in _METRICS}
        for aggregation in ("window_micro", "event_macro")
    }
    event_array = np.asarray(events, dtype=object)
    for _ in range(repeats):
        indexes = rng.integers(0, len(events), size=len(events))
        sampled = event_array[indexes]
        supported_values = _aggregate_sample(supported, sampled)
        blocked_values = _aggregate_sample(blocked, sampled)
        for aggregation in estimates:
            for metric in _METRICS:
                estimates[aggregation][metric].append(
                    blocked_values[aggregation][metric]
                    - supported_values[aggregation][metric]
                )
    output: dict[str, dict[str, dict[str, float]]] = {}
    for aggregation, metrics in estimates.items():
        output[aggregation] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            low, high = np.quantile(array, [0.025, 0.975])
            output[aggregation][metric] = {
                "bootstrap_mean": float(array.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
    return output


def _metric_view(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        aggregation: {
            metric: float(summary[aggregation][metric]) for metric in _METRICS
        }
        for aggregation in ("window_micro", "event_macro")
    }


def _deltas(
    supported: Mapping[str, Mapping[str, float]],
    blocked: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        aggregation: {
            metric: blocked[aggregation][metric] - supported[aggregation][metric]
            for metric in _METRICS
        }
        for aggregation in supported
    }


def summarize_cluster_pair(
    *,
    plan_path: Path,
    run_root: Path,
    models: tuple[str, ...],
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if plan.get("schema") != PAIR_PLAN_SCHEMA:
        raise ValueError("unsupported cluster-pair plan schema")
    folds = plan.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("cluster-pair plan has no folds")

    report_models: dict[str, Any] = {}
    for model in models:
        combined = {condition: [] for condition in PAIR_CONDITIONS}
        fold_reports: list[dict[str, Any]] = []
        for fold in folds:
            fold_index = int(fold["fold_index"])
            target_events = set(fold["target_events"])
            rows: dict[str, list[dict[str, str]]] = {}
            sources: dict[str, dict[str, str]] = {}
            for condition in PAIR_CONDITIONS:
                path = (
                    run_root
                    / "runs"
                    / model
                    / f"fold-{fold_index:02d}"
                    / condition
                    / "exports"
                    / "novel_metrics_per_window_step.csv"
                )
                rows[condition] = _read_rows(path)
                sources[condition] = {"path": str(path), "sha256": _sha256(path)}
            _validate_pair(rows["supported"], rows["blocked"], target_events)
            summaries = {
                condition: summarize_planning_rows(rows[condition])
                for condition in PAIR_CONDITIONS
            }
            for condition in PAIR_CONDITIONS:
                combined[condition].extend(rows[condition])
            views = {condition: _metric_view(summaries[condition]) for condition in PAIR_CONDITIONS}
            fold_reports.append(
                {
                    "fold_index": fold_index,
                    "cluster_id": fold["cluster_id"],
                    "anchor_event": fold["anchor_event"],
                    "target_events": sorted(target_events),
                    "sources": sources,
                    "conditions": views,
                    "blocked_minus_supported": _deltas(
                        views["supported"], views["blocked"]
                    ),
                }
            )
        combined_summaries = {
            condition: summarize_planning_rows(combined[condition])
            for condition in PAIR_CONDITIONS
        }
        combined_views = {
            condition: _metric_view(combined_summaries[condition])
            for condition in PAIR_CONDITIONS
        }
        stats = {
            condition: _event_stats(combined[condition]) for condition in PAIR_CONDITIONS
        }
        bootstrap = _paired_bootstrap(
            stats["supported"],
            stats["blocked"],
            seed=bootstrap_seed,
            repeats=bootstrap_repeats,
        )
        deltas = _deltas(combined_views["supported"], combined_views["blocked"])
        directional_gate = (
            deltas["event_macro"]["sr"] < 0
            and sum(deltas["event_macro"].values()) / len(_METRICS) < 0
        )
        report_models[model] = {
            "conditions": combined_views,
            "blocked_minus_supported": deltas,
            "paired_event_bootstrap": bootstrap,
            "target_events": len(stats["supported"]),
            "folds": fold_reports,
            "directional_gate_passed": directional_gate,
        }
    report = {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": plan["protocol_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "run_root": str(run_root),
        "bootstrap": {
            "unit": "paired target event",
            "seed": bootstrap_seed,
            "repeats": bootstrap_repeats,
            "interpretation": "sampled-target-event uncertainty only",
        },
        "models": report_models,
        "acceptance": {
            "rule": (
                "For every reported architecture, blocked-minus-supported event-macro SR is "
                "negative and the mean event-macro delta across SR/Acc/mIoU is negative."
            ),
            "passed": all(
                result["directional_gate_passed"] for result in report_models.values()
            ),
        },
        "claim_boundary": plan["claim_boundary"],
    }
    return report


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cluster-pair summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Q3.2 paired cluster results.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mlp", "transformer"])
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = summarize_cluster_pair(
        plan_path=arguments.plan,
        run_root=arguments.run_root,
        models=tuple(arguments.models),
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_repeats=arguments.bootstrap_repeats,
    )
    _write_json(arguments.output, report)
    print(
        json.dumps(
            {
                "acceptance_passed": report["acceptance"]["passed"],
                "models": {
                    model: result["blocked_minus_supported"]
                    for model, result in report["models"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
