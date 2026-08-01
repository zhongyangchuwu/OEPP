"""Fail-fast server preflight for OEPP embedding experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oepp.legacy.feature_paths import videoclip_feature_path, videoclip_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify OEPP annotations, action embeddings, feature files, and CUDA."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature", choices=("videoclip", "s3d"), default="videoclip")
    parser.add_argument(
        "--videoclip-root",
        type=Path,
        default=None,
        help="Feature directory; defaults to OEPP_VIDEOCLIP_ROOT or features/OEPP_videoclip.",
    )
    parser.add_argument(
        "--coin-s3d-root", type=Path, default=Path("/data0/wuyilu/data/COIN/full_npy")
    )
    parser.add_argument(
        "--crosstask-s3d-root", type=Path, default=Path("/data0/wuyilu/data/ori_processed_data")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--verify-feature-content",
        action="store_true",
        help="Load every feature and verify its embedding dimension.",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Validate data without requiring CUDA; never use this for server training preflight.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_path(record: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.feature == "videoclip":
        return videoclip_feature_path(
            str(record["dataset"]), str(record["vid"]), args.videoclip_root
        )
    if record["dataset"] == "COIN":
        return (
            args.coin_s3d_root
            / f"{record['task_name']}_{record['task_id_old']}_{record['vid']}.npy"
        )
    return args.crosstask_s3d_root / f"{record['task_id_old']}_{record['vid']}.npy"


def feature_content_error(path: Path, feature: str) -> str | None:
    try:
        import numpy as np

        loaded = np.load(path, mmap_mode="r", allow_pickle=True)
        array = loaded if feature == "videoclip" else loaded["frames_features"]
        expected_dimension = 768 if feature == "videoclip" else 512
        if array.ndim != 2 or array.shape[1] != expected_dimension:
            return f"{path}: expected [frames, {expected_dimension}], found {tuple(array.shape)}"
        if feature != "videoclip":
            loaded.close()
    except Exception as error:
        return f"{path}: {type(error).__name__}: {error}"
    return None


def main() -> None:
    args = parse_args()
    args.videoclip_root = videoclip_root(args.videoclip_root)
    files = {
        "train": args.data_root / "train_train_base_dataset_1.json",
        "validation": args.data_root / "train_val_base_dataset_1.json",
        "base": args.data_root / "test_base_dataset_1.json",
        "novel": args.data_root / "novel_dataset_1.json",
    }
    pools = {
        "train": load_json(args.data_root / "base_action_pool_1.json"),
        "validation": load_json(args.data_root / "base_action_pool_1.json"),
        "base": load_json(args.data_root / "base_action_pool_1.json"),
        "novel": load_json(args.data_root / "novel_action_pool_1.json"),
    }
    embedding_file = args.data_root / (
        "vc_action_feat_dict.json" if args.feature == "videoclip" else "s3d_action_feat_dict.json"
    )
    action_embeddings = load_json(embedding_file)
    report: dict[str, Any] = {
        "feature": args.feature,
        "feature_root": str(args.videoclip_root) if args.feature == "videoclip" else None,
        "cuda_check_skipped": args.skip_cuda_check,
        "splits": {},
        "errors": [],
    }

    for split_name, annotation_file in files.items():
        if not annotation_file.is_file():
            report["errors"].append(f"missing annotation file: {annotation_file}")
            continue
        records = load_json(annotation_file)
        missing_features = []
        feature_content_errors = []
        unknown_actions = []
        missing_action_embeddings = []
        for record in records:
            path = feature_path(record, args)
            if not path.is_file():
                missing_features.append(str(path))
            elif args.verify_feature_content:
                error = feature_content_error(path, args.feature)
                if error is not None:
                    feature_content_errors.append(error)
            for step in record["anno"]:
                action = str(step["action"])
                if action not in pools[split_name]:
                    unknown_actions.append(action)
                if action not in action_embeddings:
                    missing_action_embeddings.append(action)
        report["splits"][split_name] = {
            "videos": len(records),
            "missing_feature_count": len(missing_features),
            "feature_content_error_count": len(feature_content_errors),
            "unknown_action_count": len(set(unknown_actions)),
            "missing_action_embedding_count": len(set(missing_action_embeddings)),
            "missing_feature_examples": missing_features[:10],
            "feature_content_error_examples": feature_content_errors[:10],
            "unknown_action_examples": sorted(set(unknown_actions))[:10],
            "missing_action_embedding_examples": sorted(set(missing_action_embeddings))[:10],
        }
        if (
            missing_features
            or feature_content_errors
            or unknown_actions
            or missing_action_embeddings
        ):
            report["errors"].append(f"{split_name} data contract failed")

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if not torch.cuda.is_available() and not args.skip_cuda_check:
            report["errors"].append("CUDA is unavailable")
    except ImportError:
        report["torch"] = {"installed": False}
        if not args.skip_cuda_check:
            report["errors"].append("PyTorch is not installed")

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
