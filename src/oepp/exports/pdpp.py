"""Export Base/Novel embeddings from a validation-selected PDPP checkpoint."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch

from oepp.data import (
    FeatureKind,
    FeatureRoots,
    PaddingPolicy,
    Partition,
    SequenceDataset,
    SplitBundle,
)
from oepp.data.features import default_videoclip_root
from oepp.evaluation.embeddings import (
    export_split,
    render_figures,
    save_candidate_embeddings,
    write_combined_summary,
)
from oepp.training.pdpp_runtime import build_pdpp_diffusion, load_pdpp_checkpoint
from oepp.training.support import action_embedding_tensor, sha256_file, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic sampled PDPP action embeddings from a fresh checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coin-s3d-root", type=Path, default=None)
    parser.add_argument("--crosstask-s3d-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PDPP embedding export but is unavailable")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = load_pdpp_checkpoint(cli.checkpoint, device)
    train_args = Namespace(**checkpoint["args"])
    if not hasattr(train_args, "split_id"):
        raise ValueError(
            "Checkpoint predates versioned split bundles and cannot be exported by this command"
        )
    feature = FeatureKind(str(train_args.feat))
    horizon = int(train_args.horizon)
    padding = PaddingPolicy.LEFT if int(train_args.is_pad) else PaddingPolicy.NONE
    bundle = SplitBundle.load(cli.data_root, str(train_args.split_id))
    roots = FeatureRoots(
        videoclip=default_videoclip_root(),
        coin_s3d=cli.coin_s3d_root,
        crosstask_s3d=cli.crosstask_s3d_root,
    )
    model = build_pdpp_diffusion(train_args).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    output_dir = cli.output_dir or (cli.checkpoint.parent / "exports")
    output_dir.mkdir(parents=True, exist_ok=False)
    pools = {"base": bundle.action_pool("base"), "novel": bundle.action_pool("novel")}
    action_embeddings = bundle.embedding_dict(feature)
    candidate_embeddings = {
        name: action_embedding_tensor(list(pool), action_embeddings, device)
        for name, pool in pools.items()
    }
    generator = torch.Generator(device=device).manual_seed(cli.sampling_seed)

    def pdpp_predict(batch: object) -> torch.Tensor:
        start = batch[1].to(device, non_blocking=True).float()
        end = batch[2].to(device, non_blocking=True).float()
        labels = batch[7]
        current_horizon = int(labels.shape[1])
        if current_horizon != horizon:
            raise ValueError(f"Dataset horizon {current_horizon} != checkpoint horizon {horizon}")
        condition = {0: start, current_horizon - 1: end}
        with torch.inference_mode():
            output = model(condition, current_horizon, if_jump=True, generator=generator)
        start_index = int(train_args.horizon_dim) + int(train_args.class_dim)
        return output[:, :, start_index : start_index + int(train_args.action_dim)]

    datasets = {
        "base": SequenceDataset(
            bundle,
            Partition.BASE_TEST,
            feature=feature,
            horizon=horizon,
            padding=padding,
            feature_roots=roots,
        ),
        "novel": SequenceDataset(
            bundle,
            Partition.NOVEL_TEST,
            feature=feature,
            horizon=horizon,
            padding=padding,
            feature_roots=roots,
        ),
    }
    exports = {}
    for name, dataset in datasets.items():
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cli.batch_size, shuffle=False, drop_last=False
        )
        exports[name] = export_split(
            dataset=dataset,
            loader=loader,
            split_name=name,
            action_pool=list(pools[name]),
            candidate_embeddings=candidate_embeddings[name],
            predictor=pdpp_predict,
            output_dir=output_dir,
        )
    save_candidate_embeddings(output_dir, candidate_embeddings, pools)
    write_combined_summary(output_dir, exports)
    render_figures(output_dir, exports)
    write_json(
        output_dir / "run_metadata.json",
        {
            "created_at": utc_now(),
            "exporter": "oepp export-pdpp",
            "checkpoint": str(cli.checkpoint),
            "checkpoint_sha256": sha256_file(cli.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_validation_metrics": checkpoint["validation_metrics"],
            "pdpp_args": checkpoint["args"],
            "split_id": bundle.split_id,
            "split_source_hashes": dict(bundle.source_hashes),
            "device": str(device),
            "batch_size": cli.batch_size,
            "sampling_seed": cli.sampling_seed,
            "feature_roots": {"videoclip": str(roots.videoclip)},
            "splits": {name: len(dataset) for name, dataset in datasets.items()},
        },
    )


if __name__ == "__main__":
    main()
