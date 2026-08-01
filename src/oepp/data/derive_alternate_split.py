"""Derive a fully specified alternate OEPP split from recorded event assignments.

The historical source identifies Base and Novel events plus Base train/test membership, but it
has no validation partition. This command makes that completion explicit and reproducible: it
uses a hash-ranked, event-stratified holdout from the recorded Base training membership. The
output is a membership artifact; ``oepp import-alternate-split`` remains responsible for
freezing the corresponding bundle and auditing VideoCLIP coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .alternate_split import ALTERNATE_MEMBERSHIP_SCHEMA, DERIVED_EVENT_SPLIT_KIND
from .bundle import Partition, SplitBundle

DERIVATION_SCHEMA = "oepp-derived-event-split-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label.capitalize()} must be a non-empty JSON object list: {path}")
    return value


def _read_event_names(path: Path, label: str) -> set[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label}: {path}") from error
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{label.capitalize()} must be a non-empty JSON string list: {path}")
    if len(value) != len(set(value)):
        raise ValueError(f"{label.capitalize()} contains duplicate event names: {path}")
    return set(value)


def _identity(record: Mapping[str, Any], label: str) -> tuple[str, str]:
    dataset = record.get("dataset")
    vid = record.get("vid")
    if not isinstance(dataset, str) or not dataset or not isinstance(vid, str) or not vid:
        raise ValueError(f"{label} requires non-empty dataset and vid strings")
    return dataset, vid


def _event(record: Mapping[str, Any], label: str) -> tuple[str, str]:
    dataset = record.get("dataset")
    task_id = record.get("task_id")
    task_name = record.get("task_name")
    if (
        not isinstance(dataset, str)
        or not dataset
        or task_id is None
        or not isinstance(task_name, str)
        or not task_name
    ):
        raise ValueError(f"{label} requires dataset, task_id, and task_name")
    return dataset, str(task_id)


def _index_source_records(source: SplitBundle) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for partition in Partition:
        for record in source.partition_records(partition):
            identity = _identity(record, f"source {partition.value} record")
            if identity in indexed:
                raise ValueError(f"source split repeats video identity: {identity}")
            indexed[identity] = record
    return indexed


def _identities(
    records: Iterable[Mapping[str, Any]],
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    label: str,
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        identity = _identity(record, f"{label}[{index}]")
        if identity in identities:
            raise ValueError(f"{label} repeats video identity: {identity}")
        source_record = source_records.get(identity)
        if source_record is None:
            raise ValueError(
                f"{label} contains a video outside the frozen source split: {identity}"
            )
        if _event(record, f"{label}[{index}]") != _event(source_record, f"source {identity}"):
            raise ValueError(f"{label} changes the frozen event identity for {identity}")
        identities.add(identity)
    return identities


def _file_provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _validation_rank(
    identity: tuple[str, str], event: tuple[str, str], seed: int
) -> tuple[str, str, str]:
    payload = "\0".join(
        (DERIVATION_SCHEMA, str(seed), event[0], event[1], identity[0], identity[1])
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), identity[0], identity[1]


def _partition_base_training(
    base_train: set[tuple[str, str]],
    source_records: Mapping[tuple[str, str], Mapping[str, Any]],
    validation_fraction: float,
    validation_seed: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, int]]:
    by_event: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for identity in base_train:
        by_event[_event(source_records[identity], f"source {identity}")].append(identity)

    train: set[tuple[str, str]] = set()
    validation: set[tuple[str, str]] = set()
    validation_counts: dict[str, int] = {}
    for event, members in sorted(by_event.items()):
        if len(members) < 2:
            raise ValueError(
                "recorded Base training membership cannot support a non-empty event-stratified "
                f"train/validation split for event {event}"
            )
        validation_count = min(
            len(members) - 1, max(1, math.ceil(len(members) * validation_fraction))
        )
        ranked = sorted(
            members, key=lambda identity: _validation_rank(identity, event, validation_seed)
        )
        validation.update(ranked[:validation_count])
        train.update(ranked[validation_count:])
        validation_counts[f"{event[0]}:{event[1]}"] = validation_count
    return train, validation, validation_counts


def _members(identities: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"dataset": dataset, "vid": vid} for dataset, vid in sorted(identities)]


def derive_membership(
    *,
    source: SplitBundle,
    split_id: str,
    base_events_path: Path,
    novel_events_path: Path,
    base_records_path: Path,
    novel_records_path: Path,
    base_train_path: Path,
    base_test_path: Path,
    validation_fraction: float,
    validation_seed: int,
    authority: str,
    output: Path,
) -> dict[str, Any]:
    """Build an explicit, hash-provenance membership for an alternate event split."""
    if not split_id or not split_id.startswith("split-"):
        raise ValueError("split_id must be a non-empty split-* identifier")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if not authority:
        raise ValueError("authority must be non-empty")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived membership: {output}")

    source_records = _index_source_records(source)
    base_records = _read_records(base_records_path, "historical Base records")
    novel_records = _read_records(novel_records_path, "historical Novel records")
    base_train_records = _read_records(base_train_path, "historical Base training records")
    base_test_records = _read_records(base_test_path, "historical Base test records")
    base_event_names = _read_event_names(base_events_path, "historical Base event list")
    novel_event_names = _read_event_names(novel_events_path, "historical Novel event list")

    base = _identities(base_records, source_records, "historical Base records")
    novel = _identities(novel_records, source_records, "historical Novel records")
    base_train = _identities(base_train_records, source_records, "historical Base training records")
    base_test = _identities(base_test_records, source_records, "historical Base test records")
    if base.intersection(novel) or base.union(novel) != set(source_records):
        raise ValueError(
            "historical Base and Novel records must partition the frozen source corpus"
        )
    if not base_train.issubset(base) or not base_test.issubset(base):
        raise ValueError(
            "historical Base train/test records must be subsets of historical Base records"
        )
    if base_train.intersection(base_test) or base_train.union(base_test) != base:
        raise ValueError(
            "historical Base train/test records must partition historical Base records"
        )

    actual_base_names = {str(source_records[identity]["task_name"]) for identity in base}
    actual_novel_names = {str(source_records[identity]["task_name"]) for identity in novel}
    if actual_base_names != base_event_names or actual_novel_names != novel_event_names:
        raise ValueError("historical event-name lists do not match their record memberships")
    base_events = {_event(source_records[identity], f"source {identity}") for identity in base}
    novel_events = {_event(source_records[identity], f"source {identity}") for identity in novel}
    if base_events.intersection(novel_events):
        raise ValueError("historical Base and Novel event assignments overlap")

    train, validation, validation_counts = _partition_base_training(
        base_train, source_records, validation_fraction, validation_seed
    )
    membership = {
        "schema": ALTERNATE_MEMBERSHIP_SCHEMA,
        "split_id": split_id,
        "source_split": {
            "split_id": source.split_id,
            "source_hashes": dict(source.source_hashes),
        },
        "provenance": {
            "kind": DERIVED_EVENT_SPLIT_KIND,
            "authority": authority,
            "source": "Recorded alternate OEPP event membership with explicit derived validation",
            "event_assignment_rule": (
                "Historical Base/Novel event lists and historical Base train/test membership are "
                "preserved; validation is hash-ranked within each historical Base training event."
            ),
            "derivation": {
                "schema": DERIVATION_SCHEMA,
                "source_files": {
                    "base_events": _file_provenance(base_events_path),
                    "novel_events": _file_provenance(novel_events_path),
                    "base_records": _file_provenance(base_records_path),
                    "novel_records": _file_provenance(novel_records_path),
                    "base_train": _file_provenance(base_train_path),
                    "base_test": _file_provenance(base_test_path),
                },
                "validation_fraction": validation_fraction,
                "validation_seed": validation_seed,
                "validation_ranking": (
                    "sha256(schema, seed, dataset, task_id, video dataset, video id)"
                ),
                "validation_count_rule": (
                    "ceil(event_training_videos * fraction), clamped to [1, n-1]"
                ),
                "validation_event_counts": validation_counts,
            },
        },
        "partitions": {
            Partition.TRAIN.value: _members(train),
            Partition.VALIDATION.value: _members(validation),
            Partition.BASE_TEST.value: _members(base_test),
            Partition.NOVEL_TEST.value: _members(novel),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(membership, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return membership


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive an auditable alternate OEPP event split from recorded memberships."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-split-id", default="split-001")
    parser.add_argument("--split-id", default="split-002")
    parser.add_argument("--base-events", type=Path, required=True)
    parser.add_argument("--novel-events", type=Path, required=True)
    parser.add_argument("--base-records", type=Path, required=True)
    parser.add_argument("--novel-records", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-test", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=20260731)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = SplitBundle.load(arguments.data_root, arguments.source_split_id)
    membership = derive_membership(
        source=source,
        split_id=arguments.split_id,
        base_events_path=arguments.base_events,
        novel_events_path=arguments.novel_events,
        base_records_path=arguments.base_records,
        novel_records_path=arguments.novel_records,
        base_train_path=arguments.base_train,
        base_test_path=arguments.base_test,
        validation_fraction=arguments.validation_fraction,
        validation_seed=arguments.validation_seed,
        authority=arguments.authority,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "split_id": membership["split_id"],
                "partitions": {name: len(rows) for name, rows in membership["partitions"].items()},
                "validation_seed": membership["provenance"]["derivation"]["validation_seed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
