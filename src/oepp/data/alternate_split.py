"""Import and audit an authoritative alternate OEPP event split without generating membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import ActionPool, Partition, SplitBundle
from .features import FEATURE_DIMENSIONS, FeatureKind, default_videoclip_root

ALTERNATE_MEMBERSHIP_SCHEMA = "oepp-alternate-split-membership-v1"
ALTERNATE_IMPORT_AUDIT_SCHEMA = "oepp-alternate-split-import-audit-v1"
DERIVED_EVENT_SPLIT_KIND = "derived_event_split"
ALTERNATE_MEMBERSHIP_KINDS = frozenset({"authoritative_event_split", DERIVED_EVENT_SPLIT_KIND})
_SPLIT_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object")
    return value


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _identity(value: Mapping[str, Any], context: str) -> tuple[str, str]:
    dataset = value.get("dataset")
    vid = value.get("vid")
    if not isinstance(dataset, str) or not dataset or not isinstance(vid, str) or not vid:
        raise ValueError(f"{context} requires non-empty dataset and vid strings")
    return dataset, vid


def _event_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    dataset = record.get("dataset")
    task_id = record.get("task_id")
    if not isinstance(dataset, str) or not dataset or task_id is None:
        raise ValueError("frozen record requires dataset and task_id for event-split auditing")
    return dataset, str(task_id)


def _validate_split_id(split_id: object) -> str:
    if not isinstance(split_id, str) or not _SPLIT_ID_PATTERN.fullmatch(split_id):
        raise ValueError("alternate split_id must match [a-z][a-z0-9-]*")
    return split_id


def _validate_derived_provenance(provenance: Mapping[str, Any]) -> None:
    derivation = provenance.get("derivation")
    if not isinstance(derivation, Mapping):
        raise ValueError("derived event-split provenance requires a derivation object")
    if derivation.get("schema") != "oepp-derived-event-split-v1":
        raise ValueError("derived event-split provenance has an unsupported derivation schema")
    source_files = derivation.get("source_files")
    required_sources = {
        "base_events",
        "novel_events",
        "base_records",
        "novel_records",
        "base_train",
        "base_test",
    }
    if not isinstance(source_files, Mapping) or set(source_files) != required_sources:
        raise ValueError("derived event-split provenance requires complete source-file hashes")
    for name, source_file in source_files.items():
        if (
            not isinstance(source_file, Mapping)
            or not isinstance(source_file.get("path"), str)
            or not source_file["path"]
            or not isinstance(source_file.get("sha256"), str)
            or len(source_file["sha256"]) != 64
        ):
            raise ValueError(f"derived event-split provenance has an invalid {name} source file")
    fraction = derivation.get("validation_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) < 1:
        raise ValueError("derived event-split provenance has an invalid validation fraction")
    if not isinstance(derivation.get("validation_seed"), int):
        raise ValueError("derived event-split provenance has an invalid validation seed")
    for field in ("validation_ranking", "validation_count_rule"):
        if not isinstance(derivation.get(field), str) or not derivation[field]:
            raise ValueError(f"derived event-split provenance requires {field}")
    counts = derivation.get("validation_event_counts")
    if (
        not isinstance(counts, Mapping)
        or not counts
        or not all(
            isinstance(name, str) and name and isinstance(value, int) and value > 0
            for name, value in counts.items()
        )
    ):
        raise ValueError("derived event-split provenance requires positive validation event counts")


def _membership_partitions(
    membership: Mapping[str, Any], source: SplitBundle
) -> dict[Partition, tuple[tuple[str, str], ...]]:
    source_split = membership.get("source_split")
    if not isinstance(source_split, Mapping):
        raise ValueError("alternate membership requires a source_split object")
    if source_split.get("split_id") != source.split_id:
        raise ValueError(
            "alternate membership source_split.split_id does not match the source bundle"
        )
    if source_split.get("source_hashes") != dict(source.source_hashes):
        raise ValueError("alternate membership source_hashes do not match the source bundle")

    provenance = membership.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("alternate membership requires provenance")
    kind = provenance.get("kind")
    if not isinstance(kind, str) or kind not in ALTERNATE_MEMBERSHIP_KINDS:
        raise ValueError(
            "alternate membership provenance.kind must be authoritative_event_split or "
            "derived_event_split"
        )
    for field in ("authority", "source", "event_assignment_rule"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"alternate membership provenance requires non-empty {field}")
    if kind == DERIVED_EVENT_SPLIT_KIND:
        _validate_derived_provenance(provenance)

    raw_partitions = membership.get("partitions")
    if not isinstance(raw_partitions, Mapping):
        raise ValueError("alternate membership requires a partitions object")
    if set(raw_partitions) != {partition.value for partition in Partition}:
        raise ValueError(
            "alternate membership must name exactly train, validation, base_test, novel_test"
        )

    partitions: dict[Partition, tuple[tuple[str, str], ...]] = {}
    seen: set[tuple[str, str]] = set()
    for partition in Partition:
        raw_members = raw_partitions[partition.value]
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError(f"alternate membership {partition.value} must be a non-empty list")
        members: list[tuple[str, str]] = []
        for index, raw_member in enumerate(raw_members):
            if not isinstance(raw_member, Mapping):
                raise ValueError(
                    f"alternate membership {partition.value}[{index}] is not an object"
                )
            identity = _identity(raw_member, f"alternate membership {partition.value}[{index}]")
            if identity in seen:
                raise ValueError(f"alternate membership assigns a video more than once: {identity}")
            seen.add(identity)
            members.append(identity)
        partitions[partition] = tuple(members)

    source_identities = {
        _identity(record, f"source {partition.value} record")
        for partition in Partition
        for record in source.partition_records(partition)
    }
    missing = sorted(source_identities.difference(seen))
    unknown = sorted(seen.difference(source_identities))
    if missing or unknown:
        raise ValueError(
            "alternate membership must cover the frozen source corpus exactly "
            f"(missing={len(missing)}, unknown={len(unknown)})"
        )
    return partitions


def _records_for_membership(
    source: SplitBundle, membership: Mapping[Partition, tuple[tuple[str, str], ...]]
) -> dict[Partition, list[dict[str, Any]]]:
    source_records = [
        (_identity(record, f"source {partition.value} record"), record)
        for partition in Partition
        for record in source.partition_records(partition)
    ]
    selected = {partition: set(identities) for partition, identities in membership.items()}
    return {
        partition: [
            record for identity, record in source_records if identity in selected[partition]
        ]
        for partition in Partition
    }


def _event_and_action_audit(
    records: Mapping[Partition, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[ActionPool, list[str]]]:
    base_partitions = (Partition.TRAIN, Partition.VALIDATION, Partition.BASE_TEST)
    base_events = {
        _event_identity(record) for partition in base_partitions for record in records[partition]
    }
    novel_events = {_event_identity(record) for record in records[Partition.NOVEL_TEST]}
    event_overlap = sorted(base_events.intersection(novel_events))
    if event_overlap:
        raise ValueError(
            "alternate event split leaks event identities between Base and Novel: "
            f"{event_overlap[:3]}"
        )

    def actions_for(partitions: tuple[Partition, ...]) -> set[str]:
        return {
            step["action"]
            for partition in partitions
            for record in records[partition]
            for step in record["anno"]
        }

    base_actions = actions_for(base_partitions)
    novel_actions = actions_for((Partition.NOVEL_TEST,))
    return (
        {
            "base_event_count": len(base_events),
            "novel_event_count": len(novel_events),
            "base_novel_event_overlap": [],
            "base_action_count": len(base_actions),
            "novel_action_count": len(novel_actions),
            "base_novel_action_overlap": sorted(base_actions.intersection(novel_actions)),
        },
        {
            ActionPool.BASE: sorted(base_actions),
            ActionPool.NOVEL: sorted(novel_actions),
            ActionPool.TOTAL: sorted(base_actions.union(novel_actions)),
        },
    )


def _ordered_pools(
    source: SplitBundle, action_sets: Mapping[ActionPool, list[str]]
) -> dict[ActionPool, list[str]]:
    total_order = source.action_pool(ActionPool.TOTAL)
    total_actions = set(total_order)
    result: dict[ActionPool, list[str]] = {}
    for pool, actions in action_sets.items():
        unexpected = sorted(set(actions).difference(total_actions))
        if unexpected:
            raise ValueError(f"alternate {pool.value} pool has unknown actions: {unexpected[:3]}")
        result[pool] = [action for action in total_order if action in set(actions)]
    return result


def _audit_videoclip(
    records: Mapping[Partition, list[dict[str, Any]]], root: Path
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"VideoCLIP feature root does not exist: {root}")
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    checked = 0
    for partition in Partition:
        for record in records[partition]:
            dataset, vid = _identity(record, f"{partition.value} record")
            feature_path = root / f"{dataset}_{vid}.npy"
            if not feature_path.is_file():
                missing.append({"dataset": dataset, "vid": vid})
                continue
            try:
                array = np.load(feature_path, mmap_mode="r")
                valid = (
                    array.ndim == 2
                    and array.shape[0] > 0
                    and array.shape[1] == FEATURE_DIMENSIONS[FeatureKind.VIDEOCLIP]
                )
            except (EOFError, OSError, ValueError):
                valid = False
            if not valid:
                invalid.append({"dataset": dataset, "vid": vid})
                continue
            checked += 1
    return {
        "feature": FeatureKind.VIDEOCLIP.value,
        "expected_dimension": FEATURE_DIMENSIONS[FeatureKind.VIDEOCLIP],
        "checked": checked,
        "missing": missing,
        "invalid": invalid,
        "passed": not missing and not invalid,
    }


def _source_entries(
    source: SplitBundle, destination: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = _read_object(source.manifest_path, "source split manifest")

    def rel_entry(section: str, key: str) -> dict[str, Any]:
        container = source_manifest[section][key]
        source_path = (source.manifest_path.parent / container["path"]).resolve()
        relative = os.path.relpath(source_path, destination)
        return {"path": Path(relative).as_posix(), "sha256": container["sha256"]}

    task_info = source_manifest["task_info"]
    task_source = (source.manifest_path.parent / task_info["path"]).resolve()
    task_entry = {
        "path": Path(os.path.relpath(task_source, destination)).as_posix(),
        "sha256": task_info["sha256"],
    }
    embeddings = {
        feature.value: rel_entry("action_embeddings", feature.value) for feature in FeatureKind
    }
    return task_entry, embeddings


def import_alternate_split(
    data_root: Path,
    source: SplitBundle,
    membership_path: Path,
    videoclip_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Validate and freeze an externally authored alternate event split.

    This function never creates membership or shuffles data. It only maps an authoritative
    membership source onto the immutable record corpus represented by ``source``.
    """
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite alternate-split audit report: {report_path}")
    membership = _read_object(membership_path, "alternate split membership")
    if membership.get("schema") != ALTERNATE_MEMBERSHIP_SCHEMA:
        raise ValueError(f"unsupported alternate membership schema: {membership.get('schema')!r}")
    split_id = _validate_split_id(membership.get("split_id"))
    assignments = _membership_partitions(membership, source)
    records = _records_for_membership(source, assignments)
    event_audit, action_sets = _event_and_action_audit(records)
    pools = _ordered_pools(source, action_sets)
    feature_audit = _audit_videoclip(records, videoclip_root)
    if not feature_audit["passed"]:
        raise ValueError(
            "alternate split VideoCLIP audit failed "
            f"(missing={len(feature_audit['missing'])}, invalid={len(feature_audit['invalid'])})"
        )

    root = data_root.resolve()
    destination = root / "splits" / split_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite alternate split bundle: {destination}")
    try:
        report_path.resolve().relative_to(destination.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("alternate-split audit report must be outside the generated bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=destination.parent))
    membership_hash = _sha256(membership_path)
    try:
        shutil.copyfile(membership_path, staging / "authority_membership.json")
        partition_entries: dict[str, dict[str, str]] = {}
        for partition in Partition:
            filename = f"{partition.value}.json"
            output_path = staging / filename
            _write_json(output_path, records[partition])
            partition_entries[partition.value] = {"path": filename, "sha256": _sha256(output_path)}
        pool_entries: dict[str, dict[str, str]] = {}
        for pool in ActionPool:
            filename = f"{pool.value}_action_pool.json"
            output_path = staging / filename
            _write_json(output_path, pools[pool])
            pool_entries[pool.value] = {"path": filename, "sha256": _sha256(output_path)}
        task_info, action_embeddings = _source_entries(source, destination)
        manifest = {
            "schema": "oepp-split-bundle-v1",
            "split_id": split_id,
            "description": (
                "Imported derived alternate OEPP event split; membership is copied verbatim."
                if membership["provenance"]["kind"] == DERIVED_EVENT_SPLIT_KIND
                else (
                    "Imported authoritative alternate OEPP event split; "
                    "membership is copied verbatim."
                )
            ),
            "provenance": {
                **dict(membership["provenance"]),
                "source_split_id": source.split_id,
                "source_split_manifest_sha256": _sha256(source.manifest_path),
                "source_hashes": dict(source.source_hashes),
                "authority_membership": {
                    "path": "authority_membership.json",
                    "sha256": membership_hash,
                },
            },
            "partitions": partition_entries,
            "action_pools": pool_entries,
            "task_info": task_info,
            "action_embeddings": action_embeddings,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
        try:
            validated = SplitBundle.load(root, split_id)
        except BaseException:
            shutil.rmtree(destination)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report = {
        "schema": ALTERNATE_IMPORT_AUDIT_SCHEMA,
        "status": "passed",
        "split_id": split_id,
        "bundle_manifest": str(validated.manifest_path),
        "bundle_manifest_sha256": _sha256(validated.manifest_path),
        "source_split_id": source.split_id,
        "source_split_manifest_sha256": _sha256(source.manifest_path),
        "source_hashes": dict(source.source_hashes),
        "authority_membership_sha256": membership_hash,
        "partitions": {partition.value: len(records[partition]) for partition in Partition},
        "coverage": {
            "source_video_count": sum(
                len(source.partition_records(partition)) for partition in Partition
            ),
            "imported_video_count": sum(len(records[partition]) for partition in Partition),
        },
        "event_and_action_audit": event_audit,
        "videoclip_feature_audit": feature_audit,
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, report)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import an authoritative alternate OEPP event split without generating membership."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-split-id", default="split-001")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--videoclip-root", type=Path, default=default_videoclip_root())
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    source = SplitBundle.load(arguments.data_root, arguments.source_split_id)
    report = import_alternate_split(
        arguments.data_root,
        source,
        arguments.membership,
        arguments.videoclip_root,
        arguments.report,
    )
    print(
        json.dumps(
            {
                "split_id": report["split_id"],
                "status": report["status"],
                "partitions": report["partitions"],
                "videoclip_checked": report["videoclip_feature_audit"]["checked"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
