"""Fresh, validation-selected OEPP MLP/Transformer training for embedding analysis."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as functional
import yaml

from feature_paths import videoclip_root
from checkpoint_selection import is_better_direct_checkpoint
from dataset.dataset import Seq_action
from embedding_support import (
    annotation_hashes,
    canonical_json_hash,
    action_embedding_tensor,
    build_direct_model,
    load_action_embedding_dict,
    load_action_pool,
    save_direct_checkpoint,
    seed_everything,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an OEPP MLP or Transformer from fresh initialization and save state-dict checkpoints."
    )
    parser.add_argument("--config", default="attention_config.yaml", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs for timing calibration.")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"T", "split", "feature", "seed", "is_pad", "training", "model", "loss"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config {path} is missing required keys: {sorted(missing)}")
    if config["model"]["model_n"] not in {"MLP", "attention"}:
        raise ValueError("train_embeddings.py supports only model_n=MLP or model_n=attention")
    return config


def _direct_predictions(model: torch.nn.Module, frames: torch.Tensor) -> torch.Tensor:
    outputs = model(frames)
    if not isinstance(outputs, list):
        raise TypeError("Direct OEPP model must return a list of per-step embeddings")
    return torch.stack(outputs, dim=1)


def evaluate_direct(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    candidate_embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_windows = 0
    total_steps = 0
    total_correct = 0
    total_mse = 0.0
    total_iou = 0.0
    total_sr = 0
    with torch.inference_mode():
        for batch in loader:
            frames = batch[3].to(device, non_blocking=True).float()
            ground = batch[5].to(device, non_blocking=True).float()
            labels = batch[7].to(device, non_blocking=True).long()
            predicted = _direct_predictions(model, frames)
            if predicted.shape != ground.shape:
                raise ValueError(f"Predicted embedding shape {tuple(predicted.shape)} != GT shape {tuple(ground.shape)}")
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
            for truth, prediction in zip(labels.detach().cpu().tolist(), predicted_labels.detach().cpu().tolist()):
                truth_set = set(truth)
                prediction_set = set(prediction)
                total_iou += 100.0 * len(truth_set.intersection(prediction_set)) / len(truth_set.union(prediction_set))
    if total_windows == 0 or total_steps == 0:
        raise ValueError("Validation loader produced no samples")
    return {
        "sr": total_sr / total_windows,
        "acc": total_correct / total_steps,
        "mse": total_mse / total_steps,
        "miou": total_iou / total_windows,
    }




def _run_dir(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / config["model"]["model_n"] / stamp


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OEPP feature training but is unavailable")

    run_dir = _run_dir(args, config)
    run_dir.mkdir(parents=True, exist_ok=False)
    data_root = Path("data")
    seed_everything(int(config["seed"]))

    split = int(config["split"])
    horizon = int(config["T"])
    feature = str(config["feature"])
    is_pad = int(config["is_pad"])
    model_name = str(config["model"]["model_n"])
    action_embeddings = load_action_embedding_dict(data_root, feature)
    base_pool = load_action_pool(data_root, split, "base")
    base_text_embeddings = action_embedding_tensor(base_pool, action_embeddings, device)

    train_dataset = Seq_action(data_root, split, feature, horizon, is_pad, 0, 0)
    validation_dataset = Seq_action(data_root, split, feature, horizon, is_pad, 0, 3)
    batch_size = int(config["training"]["batch_size"])
    eval_batch_size = args.eval_batch_size or batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    model = build_direct_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    cross_entropy = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    provenance = {
        "created_at": utc_now(),
        "config_path": str(args.config),
        "config_hash": canonical_json_hash(config),
        "annotation_hashes": annotation_hashes(data_root, split, feature),
        "model_name": model_name,
        "feature": feature,
        "videoclip_root": str(videoclip_root()) if feature == "videoclip" else None,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "fresh_initialization": True,
        "legacy_loss_semantics": "CrossEntropyLoss receives softmax(cosine/0.1), matching OEPP/train.py",
    }
    write_json(run_dir / "run_metadata.json", provenance)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    best_metrics: Optional[dict[str, float]] = None
    best_epoch: Optional[int] = None
    metrics_path = run_dir / "training_metrics.jsonl"
    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_train_steps = 0
        for batch in train_loader:
            frames = batch[3].to(device, non_blocking=True).float()
            ground = batch[5].to(device, non_blocking=True).float()
            labels = batch[7].to(device, non_blocking=True).long()
            predicted = _direct_predictions(model, frames)
            if predicted.shape != ground.shape:
                raise ValueError(f"Predicted embedding shape {tuple(predicted.shape)} != GT shape {tuple(ground.shape)}")
            ce_total = torch.zeros((), device=device)
            mse_total = torch.zeros((), device=device)
            for step in range(horizon):
                similarity = functional.cosine_similarity(
                    predicted[:, step, :].unsqueeze(1), base_text_embeddings.unsqueeze(0), dim=2
                )
                # Preserve the original OEPP training objective exactly.
                legacy_probabilities = functional.softmax(similarity / 0.1, dim=1)
                ce_total = ce_total + cross_entropy(legacy_probabilities, labels[:, step])
                mse_total = mse_total + mse_loss(predicted[:, step, :], ground[:, step, :])
            loss = float(config["loss"]["ce_w"]) * ce_total + float(config["loss"]["mse_w"]) * mse_total
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_train_steps += 1

        validation = evaluate_direct(model, validation_loader, base_text_embeddings, device)
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
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"], "validation": validation, "best_epoch": best_epoch}))

    write_json(
        run_dir / "selection.json",
        {
            "checkpoint": "best.pt",
            "selection_rule": "maximize validation SR, then validation Acc, then minimize validation MSE, then earlier epoch",
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics or {},
            "test_sets_used_for_selection": False,
        },
    )


if __name__ == "__main__":
    main()
