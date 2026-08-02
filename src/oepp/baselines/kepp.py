"""Local-only KEPP compatibility and train-only PKG preflight.

This module does not vendor or execute the upstream KEPP implementation. It makes the
OEPP data boundaries and known compatibility blockers explicit before an adapter or
training run is approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from oepp.data import ActionPool, FeatureKind, Partition, SplitBundle, feature_dimension

UPSTREAM_KEPP_FRAME_DIMENSION = 512
UPSTREAM_KEPP_OBSERVATION_COUNT = 3
UPSTREAM_KEPP_OBSERVATION_DIMENSION = (
    UPSTREAM_KEPP_FRAME_DIMENSION * UPSTREAM_KEPP_OBSERVATION_COUNT
)


@dataclass(frozen=True)
class TrainOnlyPKG:
    """A deterministic directed action graph constructed from train records only."""

    action_to_id: Mapping[str, int]
    edge_counts: Mapping[tuple[str, str], int]
    source_record_count: int

    def as_json(self) -> dict[str, Any]:
        return {
            "action_to_id": dict(self.action_to_id),
            "edge_counts": [
                {"from": source, "to": target, "count": count}
                for (source, target), count in sorted(self.edge_counts.items())
            ],
            "source_record_count": self.source_record_count,
        }

    def summary_json(self) -> dict[str, int]:
        return {
            "action_class_count": len(self.action_to_id),
            "edge_count": len(self.edge_counts),
            "source_record_count": self.source_record_count,
        }


@dataclass(frozen=True)
class AdaptedKEPPGraph:
    """Graph tensors and immutable provenance for the native KEPP adaptation."""

    action_names: tuple[str, ...]
    node_embeddings: torch.Tensor
    normalized_adjacency: torch.Tensor
    provenance: Mapping[str, object]

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "node_embeddings": self.node_embeddings,
            "normalized_adjacency": self.normalized_adjacency,
        }


def _sha256_tensor(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


def build_adapted_kepp_graph(
    bundle: SplitBundle, feature: FeatureKind | str, device: torch.device
) -> AdaptedKEPPGraph:
    """Construct the native adaptation graph from Base training data only."""
    selected_feature = FeatureKind(feature)
    embedding_dict = bundle.embedding_dict(selected_feature)
    base_pool = tuple(bundle.action_pool(ActionPool.BASE))
    train_records = bundle.partition_records(Partition.TRAIN)
    pkg = build_train_only_pkg(train_records, base_pool)
    if tuple(sorted(pkg.action_to_id, key=pkg.action_to_id.__getitem__)) != base_pool:
        raise ValueError("Train-only KEPP graph action IDs do not match frozen Base pool order")
    node_vectors = [
        torch.as_tensor(embedding_dict[action], dtype=torch.float32) for action in base_pool
    ]
    node_embeddings = torch.cat(node_vectors, dim=0).to(device)
    if node_embeddings.ndim != 2 or node_embeddings.shape[0] != len(base_pool):
        raise ValueError("Base action embeddings do not form a valid KEPP node matrix")
    adjacency = torch.eye(len(base_pool), dtype=torch.float32, device=device)
    for (source, target), count in pkg.edge_counts.items():
        adjacency[pkg.action_to_id[source], pkg.action_to_id[target]] += count
    normalized_adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)
    pkg_json = json.dumps(pkg.as_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    provenance: dict[str, object] = {
        "source_partition": Partition.TRAIN.value,
        "base_action_pool_hash": bundle.source_hashes["pools.base"],
        "base_action_count": len(base_pool),
        "source_record_count": pkg.source_record_count,
        "edge_count": len(pkg.edge_counts),
        "pkg_sha256": hashlib.sha256(pkg_json.encode("utf-8")).hexdigest(),
        "node_embedding_sha256": _sha256_tensor(node_embeddings),
        "normalized_adjacency_sha256": _sha256_tensor(normalized_adjacency),
    }
    return AdaptedKEPPGraph(base_pool, node_embeddings, normalized_adjacency, provenance)


@dataclass(frozen=True)
class PartitionCompatibility:
    partition: str
    record_count: int
    unique_action_count: int
    actions_outside_train_class_space: tuple[str, ...]
    actions_unseen_in_train_records: tuple[str, ...]

    @property
    def is_representable_by_train_class_space(self) -> bool:
        return not self.actions_outside_train_class_space


@dataclass(frozen=True)
class KEPPPreflightReport:
    split_id: str
    source_hashes: Mapping[str, str]
    selected_feature: str
    selected_observation_dimension: int
    upstream_observation_dimension: int
    requires_feature_adapter: bool
    train_class_count: int
    base_novel_pool_overlap: tuple[str, ...]
    partitions: tuple[PartitionCompatibility, ...]
    pkg: TrainOnlyPKG
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
            "selected_observation_dimension": self.selected_observation_dimension,
            "upstream_observation_dimension": self.upstream_observation_dimension,
            "requires_feature_adapter": self.requires_feature_adapter,
            "train_class_count": self.train_class_count,
            "base_novel_pool_overlap": list(self.base_novel_pool_overlap),
            "partitions": [asdict(partition) for partition in self.partitions],
            "pkg": self.pkg.summary_json(),
            "blockers": list(self.blockers),
            "training_permitted": self.status == "ready",
        }

    def as_json(self) -> dict[str, Any]:
        payload = self.summary_json()
        payload["pkg"] = self.pkg.as_json()
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


def build_train_only_pkg(
    train_records: Iterable[Mapping[str, Any]], train_class_space: Sequence[str]
) -> TrainOnlyPKG:
    """Build the PKG from train annotations without accepting any other partition."""
    action_to_id = {action: index for index, action in enumerate(train_class_space)}
    edge_counts: Counter[tuple[str, str]] = Counter()
    record_count = 0
    for record in train_records:
        record_count += 1
        actions = _record_actions(record)
        unknown = sorted(set(actions).difference(action_to_id))
        if unknown:
            raise ValueError(f"Train record contains action outside train class space: {unknown}")
        edge_counts.update(zip(actions, actions[1:]))
    return TrainOnlyPKG(
        action_to_id=action_to_id,
        edge_counts=dict(edge_counts),
        source_record_count=record_count,
    )


def _partition_compatibility(
    partition: Partition,
    records: Sequence[Mapping[str, Any]],
    train_class_space: Mapping[str, int],
    train_observed_actions: set[str],
) -> PartitionCompatibility:
    actions = {action for record in records for action in _record_actions(record)}
    return PartitionCompatibility(
        partition=partition.value,
        record_count=len(records),
        unique_action_count=len(actions),
        actions_outside_train_class_space=tuple(sorted(actions.difference(train_class_space))),
        actions_unseen_in_train_records=tuple(sorted(actions.difference(train_observed_actions))),
    )


def inspect_kepp_compatibility(
    bundle: SplitBundle, feature: FeatureKind | str
) -> KEPPPreflightReport:
    """Assess whether frozen OEPP data can enter an unmodified closed-set KEPP pipeline."""
    selected_feature = FeatureKind(feature)
    train_class_space = bundle.action_pool(ActionPool.BASE)
    base_novel_pool_overlap = tuple(
        sorted(set(train_class_space).intersection(bundle.action_pool(ActionPool.NOVEL)))
    )
    train_records = bundle.partition_records(Partition.TRAIN)
    pkg = build_train_only_pkg(train_records, train_class_space)
    train_observed_actions = {
        action for record in train_records for action in _record_actions(record)
    }
    partitions = tuple(
        _partition_compatibility(
            partition,
            bundle.partition_records(partition),
            pkg.action_to_id,
            train_observed_actions,
        )
        for partition in Partition
    )
    selected_observation_dimension = (
        feature_dimension(selected_feature) * UPSTREAM_KEPP_OBSERVATION_COUNT
    )
    requires_feature_adapter = selected_observation_dimension != UPSTREAM_KEPP_OBSERVATION_DIMENSION
    novel = next(item for item in partitions if item.partition == Partition.NOVEL_TEST.value)
    blockers: list[str] = []
    if requires_feature_adapter:
        blockers.append(
            "Selected feature produces observation dimension "
            f"{selected_observation_dimension}; upstream KEPP expects "
            f"{UPSTREAM_KEPP_OBSERVATION_DIMENSION}. A declared feature adapter is required."
        )
    if not novel.is_representable_by_train_class_space:
        blockers.append(
            f"{len(novel.actions_outside_train_class_space)} of "
            f"{novel.unique_action_count} Novel test actions fall outside the train-only "
            "Base class space. The upstream closed-set one-hot action head cannot evaluate "
            "Novel without an explicitly approved adaptation."
        )
    blockers.append(
        "Upstream KEPP has no detected license file. Do not vendor or copy its implementation "
        "until license or permission is confirmed."
    )
    return KEPPPreflightReport(
        split_id=bundle.split_id,
        source_hashes=bundle.source_hashes,
        selected_feature=selected_feature.value,
        selected_observation_dimension=selected_observation_dimension,
        upstream_observation_dimension=UPSTREAM_KEPP_OBSERVATION_DIMENSION,
        requires_feature_adapter=requires_feature_adapter,
        train_class_count=len(train_class_space),
        base_novel_pool_overlap=base_novel_pool_overlap,
        partitions=partitions,
        pkg=pkg,
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
            "Assess frozen OEPP compatibility with KEPP without training or upstream execution."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-id", default="split-001")
    parser.add_argument(
        "--feature",
        choices=[feature.value for feature in FeatureKind],
        default="videoclip",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    report = inspect_kepp_compatibility(bundle, arguments.feature)
    payload = report.as_json()
    if arguments.output is not None:
        _write_json(arguments.output, payload)
    print(json.dumps(report.summary_json(), ensure_ascii=False, sort_keys=True))
    if report.status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
