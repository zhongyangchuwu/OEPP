"""Generate paired transferable-support and cluster-blocked OEPP split bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import ActionPool, Partition, SplitBundle
from .features import FEATURE_DIMENSIONS, FeatureKind, default_videoclip_root

PAIR_SCHEMA = "oepp-q32-cluster-pair-v1"
PAIR_PLAN_SCHEMA = "oepp-q32-cluster-pair-plan-v1"
PAIR_CONDITIONS = ("supported", "blocked")


@dataclass(frozen=True)
class PairFold:
    cluster_id: str
    cluster_events: tuple[str, ...]
    anchor_event: str
    target_events: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _identity(record: Mapping[str, Any]) -> tuple[str, str]:
    dataset = record.get("dataset")
    vid = record.get("vid")
    if not isinstance(dataset, str) or not dataset or not isinstance(vid, str) or not vid:
        raise ValueError("records require non-empty dataset and vid strings")
    return dataset, vid


def _identity_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    payload = json.dumps(sorted(_identity(record) for record in records), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_name(record: Mapping[str, Any]) -> str:
    name = record.get("task_name")
    if not isinstance(name, str) or not name:
        raise ValueError("records require a non-empty task_name")
    return name


def source_records(bundle: SplitBundle) -> tuple[dict[str, Any], ...]:
    records = tuple(
        record
        for partition in Partition
        for record in bundle.partition_records(partition)
    )
    identities = [_identity(record) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("source split repeats video identities")
    return records


def cluster_folds(bundle: SplitBundle, *, anchor_seed: int) -> tuple[PairFold, ...]:
    records = source_records(bundle)
    record_events = {_event_name(record) for record in records}
    task_events = set(bundle.task_info)
    if record_events != task_events:
        raise ValueError("task_info event names do not match the frozen source corpus")

    clusters: dict[str, list[str]] = defaultdict(list)
    for event in sorted(record_events):
        info = bundle.task_info.get(event)
        if not isinstance(info, Mapping):
            raise ValueError(f"task_info entry is invalid for {event!r}")
        cluster_id = info.get("set")
        if not isinstance(cluster_id, (int, str)) or str(cluster_id) == "":
            raise ValueError(f"task_info entry lacks a cluster id for {event!r}")
        clusters[str(cluster_id)].append(event)
    if len(clusters) < 2 or any(len(events) < 2 for events in clusters.values()):
        raise ValueError("cluster-pair evaluation requires at least two multi-event clusters")

    def anchor_rank(cluster_id: str, event: str) -> tuple[str, str]:
        payload = "\0".join((PAIR_SCHEMA, str(anchor_seed), cluster_id, event)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), event

    def cluster_rank(item: tuple[str, list[str]]) -> tuple[int, int | str]:
        cluster_id = item[0]
        try:
            return 0, int(cluster_id)
        except ValueError:
            return 1, cluster_id


    folds: list[PairFold] = []
    for cluster_id, events in sorted(clusters.items(), key=cluster_rank):
        ordered = tuple(sorted(events))
        anchor = min(ordered, key=lambda event: anchor_rank(cluster_id, event))
        targets = tuple(event for event in ordered if event != anchor)
        folds.append(
            PairFold(
                cluster_id=cluster_id,
                cluster_events=ordered,
                anchor_event=anchor,
                target_events=targets,
            )
        )
    return tuple(folds)


def _record_rank(record: Mapping[str, Any], seed: int) -> tuple[str, str, str]:
    payload = "\0".join(
        (
            PAIR_SCHEMA,
            "base-partition",
            str(seed),
            str(record.get("dataset")),
            str(record.get("task_id")),
            str(record.get("vid")),
        )
    ).encode("utf-8")
    dataset, vid = _identity(record)
    return hashlib.sha256(payload).hexdigest(), dataset, vid


def partition_base_event(
    records: Iterable[dict[str, Any]], *, seed: int
) -> dict[Partition, tuple[dict[str, Any], ...]]:
    ordered = tuple(sorted(records, key=lambda record: _record_rank(record, seed)))
    if len(ordered) < 3:
        raise ValueError("each Base event needs at least three videos for train/validation/test")
    test_count = min(len(ordered) - 2, max(1, math.ceil(len(ordered) * 0.2)))
    remaining = len(ordered) - test_count
    validation_count = min(remaining - 1, max(1, math.ceil(remaining * 0.2)))
    base_test = ordered[:test_count]
    validation = ordered[test_count : test_count + validation_count]
    train = ordered[test_count + validation_count :]
    return {
        Partition.TRAIN: train,
        Partition.VALIDATION: validation,
        Partition.BASE_TEST: base_test,
    }


def build_fold_assignments(
    bundle: SplitBundle,
    fold: PairFold,
    *,
    partition_seed: int,
) -> dict[str, dict[str, Any]]:
    records = source_records(bundle)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_event[_event_name(record)].append(record)
    all_events = set(by_event)
    cluster_events = set(fold.cluster_events)
    targets = set(fold.target_events)
    if fold.anchor_event not in cluster_events or targets != cluster_events - {fold.anchor_event}:
        raise ValueError("fold anchor/target events do not partition its cluster")

    event_partitions = {
        event: partition_base_event(event_records, seed=partition_seed)
        for event, event_records in by_event.items()
    }
    output: dict[str, dict[str, Any]] = {}
    for condition in PAIR_CONDITIONS:
        base_events = all_events - targets
        excluded_events: set[str] = set()
        if condition == "blocked":
            base_events.remove(fold.anchor_event)
            excluded_events.add(fold.anchor_event)
        assigned = {partition: [] for partition in Partition}
        for event in sorted(base_events):
            for partition in (Partition.TRAIN, Partition.VALIDATION, Partition.BASE_TEST):
                assigned[partition].extend(event_partitions[event][partition])
        for event in sorted(targets):
            assigned[Partition.NOVEL_TEST].extend(by_event[event])
        excluded = [record for event in sorted(excluded_events) for record in by_event[event]]

        assigned_ids = {
            _identity(record) for partition in Partition for record in assigned[partition]
        }
        excluded_ids = {_identity(record) for record in excluded}
        source_ids = {_identity(record) for record in records}
        if (
            assigned_ids.intersection(excluded_ids)
            or assigned_ids.union(excluded_ids) != source_ids
        ):
            raise ValueError(f"{condition} fold assignment does not partition source records")
        output[condition] = {
            "partitions": {
                partition: tuple(assigned[partition]) for partition in Partition
            },
            "excluded": tuple(excluded),
            "base_events": tuple(sorted(base_events)),
            "novel_events": fold.target_events,
        }

    supported_novel = {
        _identity(record)
        for record in output["supported"]["partitions"][Partition.NOVEL_TEST]
    }
    blocked_novel = {
        _identity(record)
        for record in output["blocked"]["partitions"][Partition.NOVEL_TEST]
    }
    if supported_novel != blocked_novel:
        raise ValueError("paired conditions must evaluate identical target videos")
    return output


def _ordered_pools(
    source: SplitBundle, partitions: Mapping[Partition, tuple[dict[str, Any], ...]]
) -> dict[ActionPool, list[str]]:
    base_actions = {
        step["action"]
        for partition in (Partition.TRAIN, Partition.VALIDATION, Partition.BASE_TEST)
        for record in partitions[partition]
        for step in record["anno"]
    }
    novel_actions = {
        step["action"]
        for record in partitions[Partition.NOVEL_TEST]
        for step in record["anno"]
    }
    source_order = source.action_pool(ActionPool.TOTAL)
    pools = {
        ActionPool.BASE: [action for action in source_order if action in base_actions],
        ActionPool.NOVEL: [action for action in source_order if action in novel_actions],
        ActionPool.TOTAL: [
            action for action in source_order if action in base_actions.union(novel_actions)
        ],
    }
    if not all(pools.values()):
        raise ValueError("cluster-pair split produced an empty action pool")
    return pools


def _source_entries(
    source: SplitBundle, destination: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(source.manifest_path.read_text(encoding="utf-8"))

    def relative_entry(section: str, key: str) -> dict[str, str]:
        entry = manifest[section][key]
        path = (source.manifest_path.parent / entry["path"]).resolve()
        return {
            "path": Path(os.path.relpath(path, destination)).as_posix(),
            "sha256": entry["sha256"],
        }

    task = manifest["task_info"]
    task_path = (source.manifest_path.parent / task["path"]).resolve()
    task_entry = {
        "path": Path(os.path.relpath(task_path, destination)).as_posix(),
        "sha256": task["sha256"],
    }
    embeddings = {
        feature.value: relative_entry("action_embeddings", feature.value)
        for feature in FeatureKind
    }
    return task_entry, embeddings


def _audit_videoclip(records: Iterable[dict[str, Any]], root: Path) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    checked = 0
    for record in records:
        dataset, vid = _identity(record)
        path = root / f"{dataset}_{vid}.npy"
        if not path.is_file():
            missing.append(f"{dataset}:{vid}")
            continue
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            valid = (
                values.ndim == 2
                and values.shape[0] > 0
                and values.shape[1] == FEATURE_DIMENSIONS[FeatureKind.VIDEOCLIP]
            )
        except (EOFError, OSError, ValueError):
            valid = False
        if valid:
            checked += 1
        else:
            invalid.append(f"{dataset}:{vid}")
    return {
        "root": str(root),
        "checked": checked,
        "missing": missing,
        "invalid": invalid,
        "passed": not missing and not invalid,
    }


def _freeze_bundle(
    *,
    data_root: Path,
    source: SplitBundle,
    split_id: str,
    fold: PairFold,
    condition: str,
    assignment: Mapping[str, Any],
    anchor_seed: int,
    partition_seed: int,
    protocol_id: str,
) -> dict[str, Any]:
    destination = data_root / "splits" / split_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite cluster-pair split: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=destination.parent))
    try:
        partitions = assignment["partitions"]
        partition_entries: dict[str, dict[str, str]] = {}
        for partition in Partition:
            path = staging / f"{partition.value}.json"
            _write_json(path, partitions[partition])
            partition_entries[partition.value] = {
                "path": path.name,
                "sha256": _sha256(path),
            }
        pools = _ordered_pools(source, partitions)
        pool_entries: dict[str, dict[str, str]] = {}
        for pool in ActionPool:
            path = staging / f"{pool.value}_action_pool.json"
            _write_json(path, pools[pool])
            pool_entries[pool.value] = {"path": path.name, "sha256": _sha256(path)}
        excluded_path = staging / "excluded_records.json"
        _write_json(excluded_path, assignment["excluded"])
        task_info, action_embeddings = _source_entries(source, destination)
        manifest = {
            "schema": "oepp-split-bundle-v1",
            "split_id": split_id,
            "description": (
                "Q3.2 paired cluster condition: a verified same-cluster anchor is available "
                "during Base fitting."
                if condition == "supported"
                else "Q3.2 paired cluster condition: the verified same-cluster anchor is excluded."
            ),
            "provenance": {
                "kind": "cluster_pair_condition",
                "schema": PAIR_SCHEMA,
                "protocol_id": protocol_id,
                "condition": condition,
                "cluster_id": fold.cluster_id,
                "cluster_events": list(fold.cluster_events),
                "anchor_event": fold.anchor_event,
                "target_events": list(fold.target_events),
                "anchor_seed": anchor_seed,
                "partition_seed": partition_seed,
                "base_partition_rule": (
                    "hash-rank videos within each Base event; allocate ceil(20%) to Base test, "
                    "then ceil(20%) of the remainder to validation, each clamped to leave train"
                ),
                "excluded_records": {
                    "path": excluded_path.name,
                    "sha256": _sha256(excluded_path),
                    "count": len(assignment["excluded"]),
                },
                "source_split_id": source.split_id,
                "source_split_manifest_sha256": _sha256(source.manifest_path),
                "source_hashes": dict(source.source_hashes),
            },
            "partitions": partition_entries,
            "action_pools": pool_entries,
            "task_info": task_info,
            "action_embeddings": action_embeddings,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
        validated = SplitBundle.load(data_root, split_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination)
        raise
    partition_counts = {
        partition.value: len(validated.partition_records(partition))
        for partition in Partition
    }
    return {
        "split_id": split_id,
        "manifest": str(validated.manifest_path),
        "manifest_sha256": _sha256(validated.manifest_path),
        "partitions": partition_counts,
        "pool_sizes": {
            pool.value: len(validated.action_pool(pool)) for pool in ActionPool
        },
        "assigned_records": sum(partition_counts.values()),
        "excluded_records": len(assignment["excluded"]),
        "novel_identity_sha256": _identity_sha256(
            validated.partition_records(Partition.NOVEL_TEST)
        ),
        "novel_pool_sha256": validated.source_hashes["pools.novel"],
        "bundle_validation_passed": True,
    }


def generate_cluster_pair_plan(
    *,
    data_root: Path,
    source_split_id: str,
    output: Path,
    protocol_id: str,
    anchor_seed: int,
    partition_seed: int,
    videoclip_root: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cluster-pair plan: {output}")
    source = SplitBundle.load(data_root, source_split_id)
    records = source_records(source)
    feature_audit = _audit_videoclip(records, videoclip_root)
    if not feature_audit["passed"]:
        raise ValueError(
            "VideoCLIP audit failed "
            f"(missing={len(feature_audit['missing'])}, invalid={len(feature_audit['invalid'])})"
        )
    folds = cluster_folds(source, anchor_seed=anchor_seed)
    split_ids = [
        f"q32-c{index:02d}-{condition}"
        for index in range(len(folds))
        for condition in PAIR_CONDITIONS
    ]
    existing = [split_id for split_id in split_ids if (data_root / "splits" / split_id).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite cluster-pair splits: {existing}")
    plan_folds: list[dict[str, Any]] = []
    created: list[Path] = []
    try:
        for index, fold in enumerate(folds):
            assignments = build_fold_assignments(source, fold, partition_seed=partition_seed)
            split_entries = {}
            for condition in PAIR_CONDITIONS:
                split_id = f"q32-c{index:02d}-{condition}"
                split_entries[condition] = _freeze_bundle(
                    data_root=data_root,
                    source=source,
                    split_id=split_id,
                    fold=fold,
                    condition=condition,
                    assignment=assignments[condition],
                    anchor_seed=anchor_seed,
                    partition_seed=partition_seed,
                    protocol_id=protocol_id,
                )
                created.append(data_root / "splits" / split_id)
            paired_fields = ("novel_identity_sha256", "novel_pool_sha256")
            if any(
                split_entries["supported"][field] != split_entries["blocked"][field]
                for field in paired_fields
            ):
                raise ValueError(
                    "paired conditions must have identical Novel records and candidate pools"
                )
            plan_folds.append(
                {
                    "fold_index": index,
                    "cluster_id": fold.cluster_id,
                    "cluster_events": list(fold.cluster_events),
                    "anchor_event": fold.anchor_event,
                    "target_events": list(fold.target_events),
                    "splits": split_entries,
                }
            )
    except BaseException:
        for destination in reversed(created):
            shutil.rmtree(destination, ignore_errors=True)
        raise
    plan = {
        "schema": PAIR_PLAN_SCHEMA,
        "protocol_id": protocol_id,
        "source_split_id": source_split_id,
        "source_split_manifest_sha256": _sha256(source.manifest_path),
        "anchor_seed": anchor_seed,
        "partition_seed": partition_seed,
        "feature_audit": feature_audit,
        "folds": plan_folds,
        "target_event_count": sum(len(fold.target_events) for fold in folds),
        "claim_boundary": (
            "Paired ablation of verified same-cluster support. The blocked condition has no "
            "same-cluster Base event; it is not an exhaustive human-verified negative relation "
            "against every cross-cluster Base event."
        ),
    }
    _write_json(output, plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired Q3.2 same-cluster-supported and cluster-blocked splits."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-split-id", default="split-001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default="oepp-q32-cluster-pair-v1")
    parser.add_argument("--anchor-seed", type=int, default=20260810)
    parser.add_argument("--partition-seed", type=int, default=20260810)
    parser.add_argument("--videoclip-root", type=Path, default=default_videoclip_root())
    arguments = parser.parse_args()
    plan = generate_cluster_pair_plan(
        data_root=arguments.data_root.resolve(),
        source_split_id=arguments.source_split_id,
        output=arguments.output,
        protocol_id=arguments.protocol_id,
        anchor_seed=arguments.anchor_seed,
        partition_seed=arguments.partition_seed,
        videoclip_root=arguments.videoclip_root.resolve(),
    )
    print(
        json.dumps(
            {
                "protocol_id": plan["protocol_id"],
                "folds": len(plan["folds"]),
                "target_events": plan["target_event_count"],
                "videoclip_checked": plan["feature_audit"]["checked"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
