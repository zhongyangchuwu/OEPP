"""Fresh, validation-selected training for registered direct OEPP planners."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from oepp.data import (
    FeatureKind,
    FeatureRoots,
    PaddingPolicy,
    Partition,
    SequenceDataset,
    SplitBundle,
)
from oepp.data.features import default_videoclip_root
from oepp.models.registry import DIRECT_MODEL_FAMILIES, build_direct_model

from .selection import is_better_direct_checkpoint
from .support import (
    action_embedding_tensor,
    canonical_json_hash,
    save_direct_checkpoint,
    seed_everything,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a registered direct OEPP planner from fresh initialization."
    )
    parser.add_argument("--config", default=Path("configs/training/transformer.yaml"), type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training.epochs for calibration."
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coin-s3d-root", type=Path, default=None)
    parser.add_argument("--crosstask-s3d-root", type=Path, default=None)
    return parser.parse_args()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config field {name} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    data = _mapping(config.get("data"), "data")
    sampling = _mapping(config.get("sampling"), "sampling")
    training = _mapping(config.get("training"), "training")
    model = _mapping(config.get("model"), "model")
    loss = _mapping(config.get("loss"), "loss")
    required_data = {"split_id", "feature", "horizon", "padding"}
    missing_data = required_data.difference(data)
    if missing_data:
        raise ValueError(f"Config {path} data is missing: {sorted(missing_data)}")
    FeatureKind(str(data["feature"]))
    if int(data["horizon"]) <= 0:
        raise ValueError("data.horizon must be positive")
    PaddingPolicy(str(data["padding"]))
    if not isinstance(data["split_id"], str) or not data["split_id"]:
        raise ValueError("data.split_id must be a non-empty string")
    if not isinstance(sampling.get("seed"), int):
        raise ValueError("sampling.seed must be an integer")
    for name in ("validation_seed", "export_seed"):
        if name in sampling and not isinstance(sampling[name], int):
            raise ValueError(f"sampling.{name} must be an integer when provided")
    for name in ("batch_size", "epochs", "lr", "weight_decay"):
        if name not in training:
            raise ValueError(f"training.{name} is required")
    if int(training["batch_size"]) <= 0 or int(training["epochs"]) <= 0:
        raise ValueError("training batch_size and epochs must be positive")
    family = model.get("family")
    if family not in DIRECT_MODEL_FAMILIES:
        raise ValueError(f"model.family must be one of {sorted(DIRECT_MODEL_FAMILIES)}")
    for name in ("ce_w", "mse_w"):
        if name not in loss:
            raise ValueError(f"loss.{name} is required")
    diversity_weight = float(loss.get("diversity_w", 0.0))
    if diversity_weight < 0.0:
        raise ValueError("loss.diversity_w must be non-negative")
    return config


def _direct_predictions(
    model: torch.nn.Module,
    frames: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    sampled_predictor = getattr(model, "predict_with_generator", None)
    outputs = (
        sampled_predictor(frames, generator=generator)
        if generator is not None and callable(sampled_predictor)
        else model(frames)
    )
    if not isinstance(outputs, list):
        raise TypeError("Direct OEPP model must return a list of per-step embeddings")
    return torch.stack(outputs, dim=1)


def _require_prediction_shape(predicted: torch.Tensor, ground: torch.Tensor) -> None:
    if predicted.shape != ground.shape:
        raise ValueError(
            f"Predicted embedding shape {tuple(predicted.shape)} != GT shape {tuple(ground.shape)}"
        )


def _sampling_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device.type).manual_seed(seed)


def evaluate_direct(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    candidate_embeddings: torch.Tensor,
    device: torch.device,
    *,
    sampling_seed: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_windows = 0
    total_steps = 0
    total_correct = 0
    total_mse = 0.0
    total_iou = 0.0
    total_sr = 0
    generator = _sampling_generator(device, sampling_seed)
    with torch.inference_mode():
        for batch in loader:
            frames = batch[3].to(device, non_blocking=True).float()
            ground = batch[5].to(device, non_blocking=True).float()
            labels = batch[7].to(device, non_blocking=True).long()
            predicted = _direct_predictions(model, frames, generator=generator)
            _require_prediction_shape(predicted, ground)
            scores = functional.cosine_similarity(
                predicted.unsqueeze(2), candidate_embeddings.unsqueeze(0).unsqueeze(0), dim=-1
            )
            predicted_labels = scores.argmax(dim=-1)
            correct = predicted_labels.eq(labels)
            total_correct += int(correct.sum().item())
            total_steps += int(labels.numel())
            total_windows += int(labels.shape[0])
            total_sr += int(correct.all(dim=1).sum().item())
            total_mse += float((predicted - ground).square().mean(dim=-1).sum().item())
            for truth, prediction in zip(
                labels.detach().cpu().tolist(), predicted_labels.detach().cpu().tolist()
            ):
                truth_set = set(truth)
                prediction_set = set(prediction)
                total_iou += (
                    100.0
                    * len(truth_set.intersection(prediction_set))
                    / len(truth_set.union(prediction_set))
                )
    if total_windows == 0 or total_steps == 0:
        raise ValueError("Validation loader produced no samples")
    return {
        "sr": total_sr / total_windows,
        "acc": total_correct / total_steps,
        "mse": total_mse / total_steps,
        "miou": total_iou / total_windows,
    }


def _run_dir(arguments: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if arguments.run_dir is not None:
        return arguments.run_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / "training" / str(config["model"]["family"]) / stamp


def _feature_roots(arguments: argparse.Namespace) -> FeatureRoots:
    return FeatureRoots(
        videoclip=default_videoclip_root(),
        coin_s3d=arguments.coin_s3d_root,
        crosstask_s3d=arguments.crosstask_s3d_root,
    )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    training = _mapping(config["training"], "training")
    data = _mapping(config["data"], "data")
    sampling = _mapping(config["sampling"], "sampling")
    if arguments.epochs is not None:
        if arguments.epochs <= 0:
            raise ValueError("--epochs must be positive")
        training["epochs"] = arguments.epochs
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OEPP feature training but is unavailable")

    run_dir = _run_dir(arguments, config)
    run_dir.mkdir(parents=True, exist_ok=False)
    seed = int(sampling["seed"])
    seed_everything(seed)
    bundle = SplitBundle.load(arguments.data_root, str(data["split_id"]))
    feature = FeatureKind(str(data["feature"]))
    horizon = int(data["horizon"])
    roots = _feature_roots(arguments)
    train_dataset = SequenceDataset(
        bundle,
        Partition.TRAIN,
        feature=feature,
        horizon=horizon,
        padding=PaddingPolicy(str(data["padding"])),
        feature_roots=roots,
    )
    validation_dataset = SequenceDataset(
        bundle,
        Partition.VALIDATION,
        feature=feature,
        horizon=horizon,
        padding=PaddingPolicy(str(data["padding"])),
        feature_roots=roots,
    )
    batch_size = int(training["batch_size"])
    eval_batch_size = arguments.eval_batch_size or batch_size
    generator = torch.Generator().manual_seed(seed)
    worker_count = int(training.get("num_workers", 0))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=worker_count,
        worker_init_fn=_seed_worker if worker_count else None,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=worker_count,
        worker_init_fn=_seed_worker if worker_count else None,
    )
    base_pool = bundle.action_pool("base")
    base_text_embeddings = action_embedding_tensor(
        list(base_pool), bundle.embedding_dict(feature), device
    )
    model = build_direct_model(
        {"feature": feature.value, "horizon": horizon, "model": config["model"]}, device
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    cross_entropy = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    diversity_weight = float(config["loss"].get("diversity_w", 0.0))
    diversity_penalty = getattr(model, "diversity_penalty", None)
    if diversity_weight and not callable(diversity_penalty):
        raise ValueError("loss.diversity_w requires a model with diversity_penalty")
    provenance = {
        "created_at": utc_now(),
        "config_path": str(arguments.config),
        "config_hash": canonical_json_hash(config),
        "split_id": bundle.split_id,
        "split_source_hashes": dict(bundle.source_hashes),
        "feature": feature.value,
        "feature_roots": {"videoclip": str(roots.videoclip)},
        "device": str(device),
        "torch_version": str(torch.__version__),
        "fresh_initialization": True,
        "sampling_seed": seed,
        "validation_sampling_seed": sampling.get("validation_seed"),
        "legacy_loss_semantics": (
            "CrossEntropyLoss receives softmax(cosine/0.1), matching OEPP's published "
            "direct baseline."
        ),
    }
    write_json(run_dir / "run_metadata.json", provenance)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    metrics_path = run_dir / "training_metrics.jsonl"
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_train_steps = 0
        for batch in train_loader:
            frames = batch[3].to(device, non_blocking=True).float()
            ground = batch[5].to(device, non_blocking=True).float()
            labels = batch[7].to(device, non_blocking=True).long()
            predicted = _direct_predictions(model, frames)
            _require_prediction_shape(predicted, ground)
            ce_total = torch.zeros((), device=device)
            mse_total = torch.zeros((), device=device)
            for step in range(horizon):
                similarity = functional.cosine_similarity(
                    predicted[:, step, :].unsqueeze(1), base_text_embeddings.unsqueeze(0), dim=2
                )
                legacy_probabilities = functional.softmax(similarity / 0.1, dim=1)
                ce_total = ce_total + cross_entropy(legacy_probabilities, labels[:, step])
                mse_total = mse_total + mse_loss(predicted[:, step, :], ground[:, step, :])
            loss = (
                float(config["loss"]["ce_w"]) * ce_total
                + float(config["loss"]["mse_w"]) * mse_total
            )
            if diversity_weight:
                assert callable(diversity_penalty)
                loss = loss + diversity_weight * diversity_penalty(frames)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_train_steps += 1
        validation = evaluate_direct(
            model,
            validation_loader,
            base_text_embeddings,
            device,
            sampling_seed=sampling.get("validation_seed"),
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_train_steps, 1),
            "validation": validation,
            "created_at": utc_now(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        save_direct_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            validation_metrics=validation,
            provenance=provenance,
        )
        if is_better_direct_checkpoint(validation, best_metrics, epoch, best_epoch):
            best_metrics = validation
            best_epoch = epoch
            save_direct_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                validation_metrics=validation,
                provenance=provenance,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": record["train_loss"],
                    "validation": validation,
                    "best_epoch": best_epoch,
                }
            )
        )
    write_json(
        run_dir / "selection.json",
        {
            "checkpoint": "best.pt",
            "selection_rule": (
                "maximize validation SR, then validation Acc, then minimize validation MSE, "
                "then earlier epoch"
            ),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics or {},
            "test_sets_used_for_selection": False,
        },
    )


if __name__ == "__main__":
    main()
