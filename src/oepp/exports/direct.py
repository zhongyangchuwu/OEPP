"""Export Base/Novel embeddings from a validation-selected direct checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from oepp.models.registry import build_direct_model
from oepp.training.support import (
    action_embedding_tensor,
    load_direct_checkpoint,
    sha256_file,
    stack_direct_outputs,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export predicted and GT OEPP embedding trajectories from a direct checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading only for a checkpoint you generated and trust.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coin-s3d-root", type=Path, default=None)
    parser.add_argument("--crosstask-s3d-root", type=Path, default=None)
    return parser.parse_args()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint config field {name} must be a mapping")
    return value


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OEPP embedding export but is unavailable")
    checkpoint = load_direct_checkpoint(
        arguments.checkpoint, device, trust_checkpoint=arguments.trust_checkpoint
    )
    config = checkpoint["config"]
    data = _mapping(config.get("data"), "data")
    model_config = _mapping(config.get("model"), "model")
    sampling = _mapping(config.get("sampling"), "sampling")
    feature = FeatureKind(str(data["feature"]))
    horizon = int(data["horizon"])
    bundle = SplitBundle.load(arguments.data_root, str(data["split_id"]))
    roots = FeatureRoots(
        videoclip=default_videoclip_root(),
        coin_s3d=arguments.coin_s3d_root,
        crosstask_s3d=arguments.crosstask_s3d_root,
    )
    model = build_direct_model(
        {"feature": feature.value, "horizon": horizon, "model": model_config}, device
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    output_dir = arguments.output_dir or (arguments.checkpoint.parent / "exports")
    output_dir.mkdir(parents=True, exist_ok=False)
    pools = {"base": bundle.action_pool("base"), "novel": bundle.action_pool("novel")}
    embeddings = bundle.embedding_dict(feature)
    candidate_embeddings = {
        name: action_embedding_tensor(list(pool), embeddings, device)
        for name, pool in pools.items()
    }
    export_seed = sampling.get("export_seed")
    if export_seed is not None and not isinstance(export_seed, int):
        raise ValueError("Checkpoint sampling.export_seed must be an integer when provided")
    export_mode = str(sampling.get("export_mode", "sample"))
    if export_mode not in {"sample", "mean"}:
        raise ValueError("Checkpoint sampling.export_mode must be 'sample' or 'mean' when provided")
    sampled_predictor = getattr(model, "predict_with_generator", None)
    mean_predictor = getattr(model, "predict_mean", None)
    generator = (
        torch.Generator(device=device.type).manual_seed(export_seed)
        if export_mode == "sample" and export_seed is not None
        else None
    )

    def direct_predict(batch: object) -> torch.Tensor:
        frames = batch[3].to(device, non_blocking=True).float()
        with torch.inference_mode():
            if export_mode == "mean":
                outputs = mean_predictor(frames) if callable(mean_predictor) else model(frames)
            else:
                outputs = (
                    sampled_predictor(frames, generator=generator)
                    if generator is not None and callable(sampled_predictor)
                    else model(frames)
                )
            return stack_direct_outputs(outputs)

    padding = PaddingPolicy(str(data["padding"]))
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
            dataset, batch_size=arguments.batch_size, shuffle=False, drop_last=False
        )
        exports[name] = export_split(
            dataset=dataset,
            loader=loader,
            split_name=name,
            action_pool=list(pools[name]),
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
            "exporter": "oepp export",
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": sha256_file(arguments.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_validation_metrics": checkpoint["validation_metrics"],
            "checkpoint_provenance": checkpoint["provenance"],
            "config": config,
            "split_id": bundle.split_id,
            "split_source_hashes": dict(bundle.source_hashes),
            "device": str(device),
            "batch_size": arguments.batch_size,
            "sampling": {
                "export_mode": export_mode,
                "export_seed": export_seed if export_mode == "sample" else None,
                "sample_count": 1 if export_mode == "sample" else 0,
                "aggregation": (
                    "single latent trajectory"
                    if export_mode == "sample"
                    else "latent mean (z=0)"
                ),
            },
            "feature_roots": {"videoclip": str(roots.videoclip)},
            "splits": {name: len(dataset) for name, dataset in datasets.items()},
        },
    )


if __name__ == "__main__":
    main()
