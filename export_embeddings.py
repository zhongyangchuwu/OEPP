"""Export continuous Base/Novel embeddings from a fresh OEPP direct-model checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from feature_paths import videoclip_root
from dataset.dataset import Seq_action
from embedding_artifacts import export_split, render_figures, save_candidate_embeddings, write_combined_summary
from embedding_support import (
    action_embedding_tensor,
    build_direct_model,
    load_action_embedding_dict,
    load_action_pool,
    load_direct_checkpoint,
    sha256_file,
    stack_direct_outputs,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export predicted and GT OEPP embedding trajectories from a direct MLP/Transformer checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OEPP embedding export but is unavailable")
    checkpoint = load_direct_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    split = int(config["split"])
    horizon = int(config["T"])
    feature = str(config["feature"])
    is_pad = int(config["is_pad"])

    model = build_direct_model(config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    output_dir = args.output_dir or (args.checkpoint.parent / "embedding_results")
    output_dir.mkdir(parents=True, exist_ok=False)
    data_root = Path("data")
    action_embedding_dict = load_action_embedding_dict(data_root, feature)
    pools = {
        "base": load_action_pool(data_root, split, "base"),
        "novel": load_action_pool(data_root, split, "novel"),
    }
    candidate_embeddings = {
        name: action_embedding_tensor(pool, action_embedding_dict, device) for name, pool in pools.items()
    }

    def direct_predict(batch: object) -> torch.Tensor:
        frames = batch[3].to(device, non_blocking=True).float()
        with torch.inference_mode():
            return stack_direct_outputs(model(frames))

    datasets = {
        "base": Seq_action(data_root, split, feature, horizon, is_pad, 0, 2),
        "novel": Seq_action(data_root, split, feature, horizon, is_pad, 0, 1),
    }
    exports = {}
    for name, dataset in datasets.items():
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        exports[name] = export_split(
            dataset=dataset,
            loader=loader,
            split_name=name,
            action_pool=pools[name],
            candidate_embeddings=candidate_embeddings[name],
            predictor=direct_predict,
            output_dir=output_dir,
        )

    save_candidate_embeddings(output_dir, candidate_embeddings, pools)
    write_combined_summary(output_dir, exports)
    render_figures(output_dir, exports)
    write_json(
        output_dir / "run_metadata.json",
        {
            "created_at": utc_now(),
            "exporter": "export_embeddings.py",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_validation_metrics": checkpoint["validation_metrics"],
            "checkpoint_provenance": checkpoint["provenance"],
            "config": config,
            "device": str(device),
            "batch_size": args.batch_size,
            "videoclip_root": str(videoclip_root()) if feature == "videoclip" else None,
            "splits": {name: len(dataset) for name, dataset in datasets.items()},
        },
    )


if __name__ == "__main__":
    main()
