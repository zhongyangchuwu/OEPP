"""Local-only P3IV compatibility preflight for the frozen OEPP protocol.

This module audits the checked-out upstream surface and OEPP contracts. It neither runs nor
copies P3IV, and it never derives a replacement split, action mapping, or feature adapter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oepp.data import ActionPool, FeatureKind, Partition, SplitBundle, feature_dimension

UPSTREAM_CROSSTASK_VISUAL_DIMENSION = 512
UPSTREAM_CROSSTASK_AUDIO_DIMENSION = 128
UPSTREAM_CROSSTASK_INPUT_DIMENSION = (
    UPSTREAM_CROSSTASK_VISUAL_DIMENSION + UPSTREAM_CROSSTASK_AUDIO_DIMENSION
)
UPSTREAM_CROSSTASK_CLASS_COUNT = 106
UPSTREAM_COIN_INPUT_DIMENSION = 512
UPSTREAM_COIN_CLASS_COUNT = 779
UPSTREAM_PREDICTION_HORIZON = 3
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM_ROOT = REPOSITORY_ROOT.parent / "upstreams" / "procedure-planning"


@dataclass(frozen=True)
class OEPPDatasetProfile:
    dataset: str
    record_count: int
    unique_action_count: int


@dataclass(frozen=True)
class P3IVPreflightReport:
    split_id: str
    source_hashes: Mapping[str, str]
    selected_feature: str
    selected_feature_dimension: int
    oepp_dataset_profiles: tuple[OEPPDatasetProfile, ...]
    base_class_count: int
    novel_class_count: int
    total_class_count: int
    novel_actions_outside_train_class_space: tuple[str, ...]
    base_novel_pool_overlap: tuple[str, ...]
    upstream_root: str
    upstream_source_paths: Mapping[str, bool]
    detected_license_files: tuple[str, ...]
    upstream_requirements: Mapping[str, Any]
    blockers: tuple[str, ...]

    @property
    def status(self) -> str:
        return "ready" if not self.blockers else "blocked"

    def summary_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "split_id": self.split_id,
            "source_hashes": dict(self.source_hashes),
            "selected_feature": self.selected_feature,
            "selected_feature_dimension": self.selected_feature_dimension,
            "oepp_dataset_profiles": [asdict(profile) for profile in self.oepp_dataset_profiles],
            "base_class_count": self.base_class_count,
            "novel_class_count": self.novel_class_count,
            "total_class_count": self.total_class_count,
            "novel_actions_outside_train_class_space_count": len(
                self.novel_actions_outside_train_class_space
            ),
            "base_novel_pool_overlap_count": len(self.base_novel_pool_overlap),
            "upstream_root": self.upstream_root,
            "upstream_source_paths": dict(self.upstream_source_paths),
            "detected_license_files": list(self.detected_license_files),
            "upstream_requirements": dict(self.upstream_requirements),
            "blockers": list(self.blockers),
            "training_permitted": self.status == "ready",
        }

    def as_json(self) -> dict[str, Any]:
        payload = self.summary_json()
        payload["novel_actions_outside_train_class_space"] = list(
            self.novel_actions_outside_train_class_space
        )
        payload["base_novel_pool_overlap"] = list(self.base_novel_pool_overlap)
        return payload


def _record_actions(record: Mapping[str, Any]) -> tuple[str, ...]:
    annotations = record.get("anno")
    if not isinstance(annotations, list):
        raise ValueError("Validated split record has non-list anno")
    actions: list[str] = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or not isinstance(annotation.get("action"), str):
            raise ValueError("Validated split record has invalid action annotation")
        actions.append(annotation["action"])
    return tuple(actions)


def _dataset_profiles(bundle: SplitBundle) -> tuple[OEPPDatasetProfile, ...]:
    counts: Counter[str] = Counter()
    actions: dict[str, set[str]] = {}
    for partition in Partition:
        for record in bundle.partition_records(partition):
            dataset = record.get("dataset")
            if not isinstance(dataset, str) or not dataset:
                raise ValueError("Validated split record has invalid dataset")
            counts[dataset] += 1
            actions.setdefault(dataset, set()).update(_record_actions(record))
    return tuple(
        OEPPDatasetProfile(dataset, counts[dataset], len(actions[dataset]))
        for dataset in sorted(counts)
    )


def _upstream_source_paths(upstream_root: Path) -> dict[str, bool]:
    return {
        "CrossTask_main.py": (upstream_root / "CrossTask_main.py").is_file(),
        "COIN_main.py": (upstream_root / "COIN_main.py").is_file(),
        "datasets": (upstream_root / "datasets").is_dir(),
        "models": (upstream_root / "models").is_dir(),
    }


def _license_files(upstream_root: Path) -> tuple[str, ...]:
    return tuple(
        candidate.name
        for candidate in sorted(upstream_root.iterdir())
        if candidate.is_file() and candidate.name.casefold() in {"license", "license.md", "copying"}
    )


def inspect_p3iv_compatibility(
    bundle: SplitBundle,
    feature: FeatureKind | str,
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT,
) -> P3IVPreflightReport:
    """Assess frozen OEPP compatibility with the unmodified upstream P3IV implementation."""
    selected_feature = FeatureKind(feature)
    selected_dimension = feature_dimension(selected_feature)
    upstream_root = upstream_root.resolve()
    source_paths = _upstream_source_paths(upstream_root)
    license_files = _license_files(upstream_root) if upstream_root.is_dir() else ()
    train_class_space = set(bundle.action_pool(ActionPool.BASE))
    novel_pool = set(bundle.action_pool(ActionPool.NOVEL))
    novel_outside = tuple(sorted(novel_pool.difference(train_class_space)))
    pool_overlap = tuple(sorted(train_class_space.intersection(novel_pool)))

    requirements = {
        "CrossTask": {
            "input_dimension": UPSTREAM_CROSSTASK_INPUT_DIMENSION,
            "visual_dimension": UPSTREAM_CROSSTASK_VISUAL_DIMENSION,
            "audio_dimension": UPSTREAM_CROSSTASK_AUDIO_DIMENSION,
            "class_count": UPSTREAM_CROSSTASK_CLASS_COUNT,
            "default_split": "upstream datasplit.pth when exist_datasplit=True",
        },
        "COIN": {
            "input_dimension": UPSTREAM_COIN_INPUT_DIMENSION,
            "visual_dimension": UPSTREAM_COIN_INPUT_DIMENSION,
            "audio_dimension": 0,
            "class_count": UPSTREAM_COIN_CLASS_COUNT,
            "default_split": "upstream train_split.pickle and test_split.pickle",
        },
        "inference": {
            "prediction_horizon": UPSTREAM_PREDICTION_HORIZON,
            "decoder": "Viterbi over a train-derived transition matrix",
            "sampling": "CrossTask inference defaults to 1500 samples; main invokes 200",
        },
    }
    blockers: list[str] = []
    if not upstream_root.is_dir():
        blockers.append(f"P3IV upstream root does not exist: {upstream_root}")
    elif not source_paths["datasets"] or not source_paths["models"]:
        missing = [name for name in ("datasets", "models") if not source_paths[name]]
        blockers.append(
            "Checked-out P3IV source lacks required tracked runtime directories "
            f"{missing}; dataset loaders and model implementation cannot be executed "
            "or fully audited."
        )
    if selected_dimension != UPSTREAM_COIN_INPUT_DIMENSION:
        blockers.append(
            f"Selected {selected_feature.value} features are {selected_dimension}-D, but upstream "
            f"COIN P3IV consumes {UPSTREAM_COIN_INPUT_DIMENSION}-D visual inputs."
        )
    if selected_dimension != UPSTREAM_CROSSTASK_INPUT_DIMENSION:
        blockers.append(
            f"Selected {selected_feature.value} features are {selected_dimension}-D, but upstream "
            f"CrossTask P3IV consumes {UPSTREAM_CROSSTASK_INPUT_DIMENSION}-D "
            f"({UPSTREAM_CROSSTASK_VISUAL_DIMENSION}-D S3D + "
            f"{UPSTREAM_CROSSTASK_AUDIO_DIMENSION}-D VGGish audio)."
        )
    blockers.append(
        "OEPP has no verified per-video COIN/CrossTask S3D archive, CrossTask VGGish audio "
        "archive, or P3IV language-embedding/action-ID provenance for this frozen split."
    )
    if len(bundle.action_pool(ActionPool.TOTAL)) != UPSTREAM_CROSSTASK_CLASS_COUNT:
        blockers.append(
            f"OEPP has {len(bundle.action_pool(ActionPool.TOTAL))} total actions, while upstream "
            f"CrossTask P3IV hard-codes {UPSTREAM_CROSSTASK_CLASS_COUNT} classes."
        )
    if len(bundle.action_pool(ActionPool.TOTAL)) != UPSTREAM_COIN_CLASS_COUNT:
        blockers.append(
            f"OEPP has {len(bundle.action_pool(ActionPool.TOTAL))} total actions, while upstream "
            f"COIN P3IV hard-codes {UPSTREAM_COIN_CLASS_COUNT} classes."
        )
    if novel_outside:
        blockers.append(
            f"{len(novel_outside)} of {len(novel_pool)} Novel actions fall outside the train-only "
            "Base action space; unmodified fixed closed-set action heads cannot produce a complete "
            "Novel evaluation."
        )
    blockers.append(
        "Upstream P3IV defaults to an upstream datasplit or random train/test split and has no "
        "OEPP validation-only checkpoint-selection path."
    )
    blockers.append(
        "P3IV sampling count and Viterbi decoding must be frozen, and the transition matrix must "
        "be rebuilt from OEPP train records only before any adapted evaluation."
    )
    blockers.append(
        "The upstream requirements pin Python 3.7, PyTorch 1.8.1+cu101, and TensorFlow 1.13.1; "
        "they are not the current OEPP locked runtime."
    )
    if not license_files:
        blockers.append(
            "Upstream P3IV has no detected license file. Do not vendor or copy its implementation "
            "until license or permission is confirmed."
        )

    return P3IVPreflightReport(
        split_id=bundle.split_id,
        source_hashes=bundle.source_hashes,
        selected_feature=selected_feature.value,
        selected_feature_dimension=selected_dimension,
        oepp_dataset_profiles=_dataset_profiles(bundle),
        base_class_count=len(train_class_space),
        novel_class_count=len(novel_pool),
        total_class_count=len(bundle.action_pool(ActionPool.TOTAL)),
        novel_actions_outside_train_class_space=novel_outside,
        base_novel_pool_overlap=pool_overlap,
        upstream_root=str(upstream_root),
        upstream_source_paths=source_paths,
        detected_license_files=license_files,
        upstream_requirements=requirements,
        blockers=tuple(blockers),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess frozen OEPP compatibility with P3IV without training or upstream execution."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument(
        "--feature",
        choices=[feature.value for feature in FeatureKind],
        default=FeatureKind.VIDEOCLIP.value,
    )
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    report = inspect_p3iv_compatibility(bundle, arguments.feature, arguments.upstream_root)
    if arguments.output is not None:
        _write_json(arguments.output, report.as_json())
    print(json.dumps(report.summary_json(), ensure_ascii=False, sort_keys=True))
    if report.status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
