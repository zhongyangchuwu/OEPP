"""Structured embedding export, summary metrics, and pre-specified OEPP figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from oepp.training.support import write_json

ROW_FIELDS = [
    "sample_id",
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
    "mse",
    "cosine",
    "gt_rank",
    "candidate_margin",
]


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int = 42, repeats: int = 2_000
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for _ in range((repeats + 199) // 200):
        indexes = rng.integers(0, values.size, size=(min(200, repeats), values.size))
        estimates.append(values[indexes].mean(axis=1))
    boot = np.concatenate(estimates)[:repeats]
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_split(
    *,
    dataset: Any,
    loader: Iterable[Any],
    split_name: str,
    action_pool: list[str],
    candidate_embeddings: torch.Tensor,
    predictor: Callable[[Any], torch.Tensor],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export all continuous predictions for one deterministic, non-shuffled split."""
    root = Path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    predicted_labels_out: list[np.ndarray] = []
    mse_out: list[np.ndarray] = []
    cosine_out: list[np.ndarray] = []
    rank_out: list[np.ndarray] = []
    margin_out: list[np.ndarray] = []
    correct_out: list[np.ndarray] = []
    padded_out: list[np.ndarray] = []
    offset = 0

    for batch in loader:
        predicted = predictor(batch)
        ground = batch[5].to(predicted.device, non_blocking=True).float()
        labels = batch[7].to(predicted.device, non_blocking=True).long()
        if predicted.ndim != 3 or ground.shape != predicted.shape:
            raise ValueError(
                f"Expected predicted and GT embeddings of shape [batch, T, D], got "
                f"{tuple(predicted.shape)} and {tuple(ground.shape)}"
            )
        if labels.shape != predicted.shape[:2]:
            raise ValueError(
                f"Label shape {tuple(labels.shape)} does not match prediction shape "
                f"{tuple(predicted.shape)}"
            )
        if labels.min().item() < 0 or labels.max().item() >= len(action_pool):
            raise ValueError("Dataset emitted labels outside the selected candidate pool")

        scores = functional.cosine_similarity(
            predicted.unsqueeze(2), candidate_embeddings.unsqueeze(0).unsqueeze(0), dim=-1
        )
        predicted_labels = scores.argmax(dim=-1)
        gt_scores = scores.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        non_gt_scores = scores.clone()
        non_gt_scores.scatter_(2, labels.unsqueeze(-1), float("-inf"))
        margins = gt_scores - non_gt_scores.max(dim=-1).values
        ranks = (scores > gt_scores.unsqueeze(-1)).sum(dim=-1) + 1
        mse = (predicted - ground).square().mean(dim=-1)
        cosine = functional.cosine_similarity(predicted, ground, dim=-1)
        correct = predicted_labels.eq(labels)

        predictions.append(predicted.detach().cpu().numpy().astype(np.float32, copy=False))
        ground_truth.append(ground.detach().cpu().numpy().astype(np.float32, copy=False))
        labels_out.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
        predicted_labels_out.append(
            predicted_labels.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        mse_out.append(mse.detach().cpu().numpy().astype(np.float32, copy=False))
        cosine_out.append(cosine.detach().cpu().numpy().astype(np.float32, copy=False))
        rank_out.append(ranks.detach().cpu().numpy().astype(np.int64, copy=False))
        margin_out.append(margins.detach().cpu().numpy().astype(np.float32, copy=False))
        correct_out.append(correct.detach().cpu().numpy().astype(bool, copy=False))

        batch_size, horizon = labels.shape
        for local_index in range(batch_size):
            metadata = dataset.metadata_at(offset + local_index)
            expected_actions = [
                action_pool[int(label)] for label in labels[local_index].detach().cpu().tolist()
            ]
            if metadata["actions"] != expected_actions:
                raise ValueError(
                    f"Dataset metadata/action-label mismatch for {metadata['sample_id']}: "
                    f"{metadata['actions']} != {expected_actions}"
                )
            for step in range(horizon):
                rows.append(
                    {
                        **{key: metadata[key] for key in ROW_FIELDS if key in metadata},
                        "step_index": step,
                        "gt_action": action_pool[int(labels[local_index, step])],
                        "gt_label": int(labels[local_index, step]),
                        "predicted_action": action_pool[int(predicted_labels[local_index, step])],
                        "predicted_label": int(predicted_labels[local_index, step]),
                        "correct": int(correct[local_index, step]),
                        "mse": float(mse[local_index, step]),
                        "cosine": float(cosine[local_index, step]),
                        "gt_rank": int(ranks[local_index, step]),
                        "candidate_margin": float(margins[local_index, step]),
                    }
                )
        padded_out.append(
            np.repeat(
                np.asarray(
                    [
                        bool(dataset.metadata_at(offset + index)["is_padded"])
                        for index in range(batch_size)
                    ]
                )[:, None],
                horizon,
                axis=1,
            )
        )
        offset += batch_size

    if offset != len(dataset):
        raise ValueError(f"Exporter consumed {offset} samples, expected {len(dataset)}")

    arrays = {
        "pred_embeddings": np.concatenate(predictions, axis=0),
        "gt_embeddings": np.concatenate(ground_truth, axis=0),
        "gt_labels": np.concatenate(labels_out, axis=0),
        "predicted_labels": np.concatenate(predicted_labels_out, axis=0),
        "mse": np.concatenate(mse_out, axis=0),
        "cosine": np.concatenate(cosine_out, axis=0),
        "gt_rank": np.concatenate(rank_out, axis=0),
        "candidate_margin": np.concatenate(margin_out, axis=0),
        "correct": np.concatenate(correct_out, axis=0),
        "is_padded": np.concatenate(padded_out, axis=0),
    }
    np.savez_compressed(raw_dir / f"{split_name}_embeddings.npz", **arrays)
    _write_rows(root / f"{split_name}_metrics_per_window_step.csv", rows)

    summary = summarize_rows(rows, split_name, arrays)
    write_json(root / f"{split_name}_summary.json", summary)
    return {"rows": rows, "arrays": arrays, "summary": summary}


def summarize_rows(
    rows: list[dict[str, Any]], split_name: str, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for step in sorted({int(row["step_index"]) for row in rows}):
        step_rows = [row for row in rows if int(row["step_index"]) == step]
        record: dict[str, Any] = {
            "split": split_name,
            "step_index": step,
            "samples": len(step_rows),
        }
        for metric in ("mse", "cosine", "candidate_margin", "correct", "gt_rank"):
            values = np.asarray([float(row[metric]) for row in step_rows])
            mean, low, high = _bootstrap_mean_ci(values, seed=42 + step)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        summary_rows.append(record)
    all_values = {
        metric: np.asarray([float(row[metric]) for row in rows])
        for metric in ("mse", "cosine", "candidate_margin", "correct", "gt_rank")
    }
    overall = {
        "split": split_name,
        "samples": int(arrays["mse"].shape[0]),
        "steps": int(arrays["mse"].shape[1]),
    }
    for metric, values in all_values.items():
        mean, low, high = _bootstrap_mean_ci(values)
        overall[f"{metric}_mean"] = mean
        overall[f"{metric}_ci_low"] = low
        overall[f"{metric}_ci_high"] = high
    overall["zero_vector_mse"] = float(np.mean(arrays["gt_embeddings"] ** 2))
    return {"overall": overall, "by_step": summary_rows}


def write_combined_summary(
    output_dir: str | Path, exports: Mapping[str, Mapping[str, Any]]
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = [entry for exported in exports.values() for entry in exported["summary"]["by_step"]]
    with (root / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        root / "summary_metrics.json",
        {name: exported["summary"] for name, exported in exports.items()},
    )


def _group(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def _plot_mean_ci(
    axis: Any, rows: list[dict[str, Any]], metric: str, label: str, color: str
) -> None:
    grouped = _group(rows, "step_index")
    steps = sorted(int(step) for step in grouped)
    means, lows, highs = [], [], []
    for step in steps:
        values = np.asarray([float(row[metric]) for row in grouped[step]])
        mean, low, high = _bootstrap_mean_ci(values, seed=42 + step)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    axis.errorbar(
        steps,
        means,
        yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
        marker="o",
        capsize=3,
        label=label,
        color=color,
    )


def render_figures(output_dir: str | Path, exports: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    base_rows = exports["base"]["rows"]
    novel_rows = exports["novel"]["rows"]

    for metric, filename, ylabel, zero_line in (
        ("cosine", "cosine_by_step.png", "cosine(predicted embedding, GT text embedding)", None),
        ("mse", "mse_by_step.png", "raw MSE", None),
        (
            "candidate_margin",
            "candidate_margin_by_step.png",
            "GT cosine − strongest non-GT cosine",
            0.0,
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        _plot_mean_ci(axis, base_rows, metric, "Base", "#1f77b4")
        _plot_mean_ci(axis, novel_rows, metric, "Novel", "#ff7f0e")
        if metric == "mse":
            for name, exported, color in (
                ("Base zero-vector", exports["base"], "#1f77b4"),
                ("Novel zero-vector", exports["novel"], "#ff7f0e"),
            ):
                axis.axhline(
                    exported["summary"]["overall"]["zero_vector_mse"],
                    linestyle="--",
                    color=color,
                    alpha=0.6,
                    label=name,
                )
        if zero_line is not None:
            axis.axhline(zero_line, linestyle="--", color="black", linewidth=1)
        axis.set_xlabel("Procedure step")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=200)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels, distributions, colors = [], [], []
    for split_name, rows, color in (
        ("Base", base_rows, "#1f77b4"),
        ("Novel", novel_rows, "#ff7f0e"),
    ):
        for step in sorted({int(row["step_index"]) for row in rows}):
            labels.append(f"{split_name}\nstep {step + 1}")
            distributions.append(
                [float(row["cosine"]) for row in rows if int(row["step_index"]) == step]
            )
            colors.append(color)
    boxes = axis.boxplot(distributions, labels=labels, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axis.set_ylabel("cosine(predicted embedding, GT text embedding)")
    figure.tight_layout()
    figure.savefig(figure_dir / "cosine_distribution_by_step.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for axis, metric, title in zip(
        axes.ravel(),
        ("cosine", "mse", "correct", "gt_rank"),
        ("Mean cosine", "Mean raw MSE", "Top-1 accuracy", "Mean GT rank"),
    ):
        values = [
            exports["base"]["summary"]["overall"][f"{metric}_mean"],
            exports["novel"]["summary"]["overall"][f"{metric}_mean"],
        ]
        axis.bar((0, 1), values, color=("#1f77b4", "#ff7f0e"))
        axis.set_xticks((0, 1), ("Base", "Novel"))
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(figure_dir / "base_vs_novel.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for axis, metric, title in zip(
        axes, ("cosine", "candidate_margin"), ("Cosine", "Candidate margin")
    ):
        distributions, labels = [], []
        for name, rows in (
            ("correct", [row for row in base_rows + novel_rows if int(row["correct"]) == 1]),
            ("wrong", [row for row in base_rows + novel_rows if int(row["correct"]) == 0]),
        ):
            if rows:
                distributions.append([float(row[metric]) for row in rows])
                labels.append(name)
        if distributions:
            axis.boxplot(distributions, labels=labels)
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(figure_dir / "correct_vs_wrong.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    labels, values, colors = [], [], []
    for split_name, rows, color in (
        ("Base", base_rows, "#1f77b4"),
        ("Novel", novel_rows, "#ff7f0e"),
    ):
        for category, subset in (
            ("all", rows),
            ("non-padded", [row for row in rows if not bool(row["is_padded"])]),
        ):
            labels.append(f"{split_name}\n{category}")
            values.append(
                float(np.mean([float(row["cosine"]) for row in subset])) if subset else float("nan")
            )
            colors.append(color)
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.set_xticks(np.arange(len(values)), labels)
    axis.set_ylabel("mean cosine")
    figure.tight_layout()
    figure.savefig(figure_dir / "padding_sensitivity.png", dpi=200)
    plt.close(figure)


def save_candidate_embeddings(
    output_dir: str | Path, embeddings: Mapping[str, torch.Tensor], pools: Mapping[str, list[str]]
) -> None:
    root = Path(output_dir) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for split_name, tensor in embeddings.items():
        payload[f"{split_name}_embeddings"] = (
            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        payload[f"{split_name}_actions"] = np.asarray(pools[split_name], dtype=str)
    np.savez_compressed(root / "candidate_embeddings.npz", **payload)
