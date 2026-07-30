"""Versioned OEPP split bundles with source-hash and leakage validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .features import FeatureKind, feature_dimension


class Partition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    BASE_TEST = "base_test"
    NOVEL_TEST = "novel_test"


class ActionPool(str, Enum):
    BASE = "base"
    NOVEL = "novel"
    TOTAL = "total"


BUNDLE_SCHEMA = "oepp-split-bundle-v1"


@dataclass(frozen=True)
class SourceFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SplitBundle:
    """Immutable, validated view of one OEPP split definition."""

    split_id: str
    manifest_path: Path
    provenance: Mapping[str, Any]
    records: Mapping[Partition, tuple[dict[str, Any], ...]]
    pools: Mapping[ActionPool, tuple[str, ...]]
    task_info: Mapping[str, Any]
    action_embeddings: Mapping[FeatureKind, Mapping[str, Any]]
    source_hashes: Mapping[str, str]

    @classmethod
    def load(cls, data_root: Path, split_id: str) -> SplitBundle:
        root = data_root.resolve()
        manifest_path = root / "splits" / split_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
        manifest = _load_object(manifest_path, "split manifest")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"Unsupported split schema: {manifest.get('schema')!r}")
        if manifest.get("split_id") != split_id:
            raise ValueError("Split manifest split_id does not match its requested identifier")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Split manifest provenance must be an object")

        records: dict[Partition, tuple[dict[str, Any], ...]] = {}
        pools: dict[ActionPool, tuple[str, ...]] = {}
        embeddings: dict[FeatureKind, Mapping[str, Any]] = {}
        hashes: dict[str, str] = {}
        for partition in Partition:
            entry = _source_entry(
                manifest, "partitions", partition.value, manifest_path.parent, root
            )
            records[partition] = _load_records(entry.path, partition)
            hashes[f"records.{partition.value}"] = entry.sha256
        for pool in ActionPool:
            entry = _source_entry(manifest, "action_pools", pool.value, manifest_path.parent, root)
            pools[pool] = _load_action_pool(entry.path, pool)
            hashes[f"pools.{pool.value}"] = entry.sha256
        task_entry = _source_entry(manifest, "task_info", None, manifest_path.parent, root)
        task_info = _load_object(task_entry.path, "task info")
        hashes["task_info"] = task_entry.sha256
        for feature in FeatureKind:
            entry = _source_entry(
                manifest, "action_embeddings", feature.value, manifest_path.parent, root
            )
            embeddings[feature] = _load_embeddings(entry.path, feature)
            hashes[f"action_embeddings.{feature.value}"] = entry.sha256

        _validate_records(records, pools)
        _validate_pools(pools)
        _validate_embeddings(embeddings, pools)
        return cls(
            split_id=split_id,
            manifest_path=manifest_path,
            provenance=provenance,
            records=records,
            pools=pools,
            task_info=task_info,
            action_embeddings=embeddings,
            source_hashes=hashes,
        )

    def partition_records(self, partition: Partition | str) -> tuple[dict[str, Any], ...]:
        return self.records[Partition(partition)]

    def action_pool(self, pool: ActionPool | str) -> tuple[str, ...]:
        return self.pools[ActionPool(pool)]

    def action_pool_for(
        self, partition: Partition | str, *, total: bool = False
    ) -> tuple[str, ...]:
        if total:
            return self.action_pool(ActionPool.TOTAL)
        selected = Partition(partition)
        pool = ActionPool.NOVEL if selected is Partition.NOVEL_TEST else ActionPool.BASE
        return self.action_pool(pool)

    def embedding_dict(self, feature: FeatureKind | str) -> Mapping[str, Any]:
        return self.action_embeddings[FeatureKind(feature)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return value


def _resolve_source(path_text: Any, bundle_root: Path, data_root: Path) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("Bundle source path must be a non-empty relative string")
    candidate = bundle_root / path_text
    resolved = candidate.resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"Bundle source path escapes data root: {path_text}") from error
    return resolved


def _source_entry(
    manifest: Mapping[str, Any],
    section: str,
    key: str | None,
    bundle_root: Path,
    data_root: Path,
) -> SourceFile:
    container = manifest.get(section)
    if key is not None:
        if not isinstance(container, dict):
            raise ValueError(f"Manifest {section} must be an object")
        container = container.get(key)
    if not isinstance(container, dict):
        label = f"{section}.{key}" if key else section
        raise ValueError(f"Manifest source entry missing: {label}")
    path = _resolve_source(container.get("path"), bundle_root, data_root)
    expected_hash = container.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Manifest source hash is invalid for {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest source does not exist: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Source hash mismatch for {path}: expected {expected_hash}, got {actual_hash}"
        )
    return SourceFile(path=path, sha256=actual_hash)


def _load_records(path: Path, partition: Partition) -> tuple[dict[str, Any], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON records for {partition.value}: {path}") from error
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise ValueError(f"Records for {partition.value} must be a JSON object list")
    return tuple(value)


def _load_action_pool(path: Path, pool: ActionPool) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON action pool for {pool.value}: {path}") from error
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(action, str) and action for action in value)
    ):
        raise ValueError(f"Action pool {pool.value} must be a non-empty JSON string list")
    if len(value) != len(set(value)):
        raise ValueError(f"Action pool {pool.value} contains duplicate strings")
    return tuple(value)


def _load_embeddings(path: Path, feature: FeatureKind) -> Mapping[str, Any]:
    value = _load_object(path, f"{feature.value} action embeddings")
    return value


def _validate_records(
    records: Mapping[Partition, tuple[dict[str, Any], ...]],
    pools: Mapping[ActionPool, tuple[str, ...]],
) -> None:
    seen_videos: set[tuple[str, str]] = set()
    for partition, partition_records in records.items():
        expected_pool = set(
            pools[ActionPool.NOVEL if partition is Partition.NOVEL_TEST else ActionPool.BASE]
        )
        for record in partition_records:
            dataset = record.get("dataset")
            vid = record.get("vid")
            annotations = record.get("anno")
            if not isinstance(dataset, str) or not dataset or not isinstance(vid, str) or not vid:
                raise ValueError(f"{partition.value} record requires non-empty dataset and vid")
            identity = (dataset, vid)
            if identity in seen_videos:
                raise ValueError(f"Video appears in more than one partition: {identity}")
            seen_videos.add(identity)
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"{partition.value} record {identity} requires non-empty anno")
            actions = []
            for step in annotations:
                if not isinstance(step, dict) or not isinstance(step.get("action"), str):
                    raise ValueError(
                        f"{partition.value} record {identity} has invalid annotation action"
                    )
                segment = step.get("segment")
                if not isinstance(segment, list) or len(segment) != 2:
                    raise ValueError(
                        f"{partition.value} record {identity} has invalid annotation segment"
                    )
                actions.append(step["action"])
            missing_actions = sorted(set(actions).difference(expected_pool))
            if missing_actions:
                raise ValueError(
                    f"{partition.value} record {identity} has actions outside its pool: "
                    f"{missing_actions}"
                )


def _validate_pools(pools: Mapping[ActionPool, tuple[str, ...]]) -> None:
    expected_total = set(pools[ActionPool.BASE]).union(pools[ActionPool.NOVEL])
    if set(pools[ActionPool.TOTAL]) != expected_total:
        raise ValueError("Total action pool must equal the union of Base and Novel pools")


def _has_embedding_dimension(value: Any, dimension: int) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == dimension:
        return True
    return len(value) == 1 and isinstance(value[0], list) and len(value[0]) == dimension


def _validate_embeddings(
    embeddings: Mapping[FeatureKind, Mapping[str, Any]], pools: Mapping[ActionPool, tuple[str, ...]]
) -> None:
    all_actions = set(pools[ActionPool.TOTAL])
    for feature, values in embeddings.items():
        missing = sorted(all_actions.difference(values))
        if missing:
            raise ValueError(
                f"{feature.value} action embeddings are missing pool actions: {missing}"
            )
        dimension = feature_dimension(feature)
        invalid = [
            action
            for action in all_actions
            if not _has_embedding_dimension(values[action], dimension)
        ]
        if invalid:
            raise ValueError(
                f"{feature.value} action embeddings have invalid dimensions for: {invalid}"
            )
