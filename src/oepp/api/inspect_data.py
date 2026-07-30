from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import REPOSITORY_ROOT, load_json, utc_now, write_json

SPLIT_FILES = {
    "train": "train_train_base_dataset_1.json",
    "validation": "train_val_base_dataset_1.json",
    "base": "test_base_dataset_1.json",
    "novel": "novel_dataset_1.json",
}
POOL_FILES = {
    "base": "base_action_pool_1.json",
    "novel": "novel_action_pool_1.json",
    "total": "total_action_pool.json",
}
PAPER_TABLE_II = {
    "train": {"videos": 1285, "segments": 5833},
    "validation": {"videos": 337, "segments": 1479},
    "base": {"videos": 416, "segments": 1888},
    "novel": {"videos": 733, "segments": 3010},
}
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


def _window_count(records: list[dict[str, Any]], horizon: int) -> int:
    """Count protocol windows, including one left-padded window for short sequences."""
    return sum(
        max(1, len(record["anno"]) - horizon + 1) if record["anno"] else 0 for record in records
    )


def _duplicates(values: list[str]) -> dict[str, int]:
    return {value: count for value, count in Counter(values).items() if count > 1}


def _split_audit(
    name: str,
    records: list[dict[str, Any]],
    expected_pool: list[str],
) -> dict[str, Any]:
    missing_actions = [
        {"dataset": record["dataset"], "vid": record["vid"], "action": step["action"]}
        for record in records
        for step in record["anno"]
        if step["action"] not in expected_pool
    ]
    raw_vid_duplicates = _duplicates([record["vid"] for record in records])
    identity_duplicates = _duplicates(
        [f"{record['dataset']}::{record['vid']}" for record in records]
    )
    declared_length_mismatches = [
        {
            "dataset": record["dataset"],
            "vid": record["vid"],
            "length": record["length"],
            "anno_length": len(record["anno"]),
        }
        for record in records
        if record["length"] != len(record["anno"])
    ]
    fields = sorted({field for record in records for field in record})
    return {
        "file": SPLIT_FILES[name],
        "videos": len(records),
        "segments": sum(len(record["anno"]) for record in records),
        "events": len({record["task_name"] for record in records}),
        "domains": len({record["domain"] for record in records}),
        "record_fields": fields,
        "action_fields": sorted(
            {field for record in records for step in record["anno"] for field in step}
        ),
        "declared_length_distribution": dict(
            sorted(Counter(record["length"] for record in records).items())
        ),
        "sequence_windows": {"T3": _window_count(records, 3), "T4": _window_count(records, 4)},
        "declared_length_mismatches": declared_length_mismatches,
        "raw_vid_duplicates": raw_vid_duplicates,
        "dataset_vid_duplicates": identity_duplicates,
        "ground_truth_actions_missing_from_protocol_pool": missing_actions,
        "paper_table_ii_match": {
            metric: value == PAPER_TABLE_II[name][metric]
            for metric, value in {
                "videos": len(records),
                "segments": sum(len(record["anno"]) for record in records),
            }.items()
        },
    }


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# OEPP API Evaluation Data Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        f"Data root: `{audit['data_root']}`",
        "",
        "## Paper consistency",
        "",
        "| Split | Videos | Segments | Table II match |",
        "|---|---:|---:|---|",
    ]
    for name, details in audit["splits"].items():
        matches = details["paper_table_ii_match"]
        status = "yes" if all(matches.values()) else "no"
        lines.append(f"| {name} | {details['videos']} | {details['segments']} | {status} |")
    lines.extend(["", "## Sequence windows", "", "| Split | T=3 | T=4 |", "|---|---:|---:|"])
    for name, details in audit["splits"].items():
        windows = details["sequence_windows"]
        lines.append(f"| {name} | {windows['T3']} | {windows['T4']} |")
    lines.extend(
        [
            "",
            "## Action pools",
            "",
            "| Pool | Entries | Unique | Duplicate entries |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, details in audit["action_pools"].items():
        lines.append(
            f"| {name} | {details['entries']} | {details['unique_entries']} | "
            f"{details['duplicate_entries']} |"
        )
    lines.extend(
        [
            "",
            f"Base/Novel string overlap: **{audit['action_pool_overlap']['base_novel_overlap']}**.",
            "",
            "## Integrity findings",
            "",
        ]
    )
    for name, details in audit["splits"].items():
        missing_actions = len(details["ground_truth_actions_missing_from_protocol_pool"])
        length_mismatches = len(details["declared_length_mismatches"])
        identity_duplicates = len(details["dataset_vid_duplicates"])
        lines.append(
            f"- **{name}:** {missing_actions} ground-truth actions outside its protocol pool; "
            f"{length_mismatches} declared-length mismatches; "
            f"{identity_duplicates} duplicate `(dataset, vid)` identities."
        )
    assets = audit["visual_assets"]
    image_field_count = len(assets["image_or_frame_fields"])
    lines.extend(
        [
            "",
            "## Visual asset readiness",
            "",
            f"- Image files under data root: **{assets['images_under_data_root']}**.",
            f"- Image files under repository root: **{assets['images_under_repository_root']}**.",
            f"- Annotation fields containing image/frame paths: **{image_field_count}**.",
            f"- Visual manifest ready: **{str(assets['manifest_ready']).lower()}**.",
            "",
            "The current repository cannot produce a valid visual API manifest: annotations "
            "identify timestamps and action segments, but do not identify raw frame files. "
            "Supply a source-backed observation index before running `build_manifest.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit(data_root: Path) -> dict[str, Any]:
    records_by_split = {
        name: load_json(data_root / filename) for name, filename in SPLIT_FILES.items()
    }
    if not all(isinstance(records, list) for records in records_by_split.values()):
        raise ValueError("Every OEPP split file must contain a JSON list.")
    pools = {name: load_json(data_root / filename) for name, filename in POOL_FILES.items()}
    if not all(
        isinstance(pool, list) and all(isinstance(action, str) for action in pool)
        for pool in pools.values()
    ):
        raise ValueError("Every OEPP action-pool file must contain a JSON string list.")

    split_audits = {
        name: _split_audit(name, records, pools["novel"] if name == "novel" else pools["base"])
        for name, records in records_by_split.items()
    }
    data_images = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    repository_images = [
        path
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "oepp_api_eval" not in path.parts
    ]
    record_fields = {
        field.lower()
        for records in records_by_split.values()
        for record in records
        for field in record
    }
    image_or_frame_fields = sorted(
        field for field in record_fields if "image" in field or "frame" in field
    )
    return {
        "generated_at": utc_now(),
        "data_root": str(data_root.resolve()),
        "files": [
            {"path": filename, "bytes": (data_root / filename).stat().st_size}
            for filename in sorted(path.name for path in data_root.glob("*.json"))
        ],
        "splits": split_audits,
        "action_pools": {
            name: {
                "file": POOL_FILES[name],
                "entries": len(pool),
                "unique_entries": len(set(pool)),
                "duplicate_entries": len(pool) - len(set(pool)),
            }
            for name, pool in pools.items()
        },
        "action_pool_overlap": {
            "base_novel_overlap": len(set(pools["base"]) & set(pools["novel"])),
            "base_novel_union": len(set(pools["base"]) | set(pools["novel"])),
        },
        "visual_assets": {
            "images_under_data_root": len(data_images),
            "images_under_repository_root": len(repository_images),
            "image_or_frame_fields": image_or_frame_fields,
            "manifest_ready": bool(data_images and image_or_frame_fields),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit OEPP annotation and action-pool data without mutation."
    )
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=Path("data_audit"))
    arguments = parser.parse_args()
    audit_report = audit(arguments.data_root)
    output_dir = arguments.output_dir
    write_json(output_dir / "data_audit.json", audit_report)
    (output_dir / "data_audit.md").write_text(_render_markdown(audit_report), encoding="utf-8")
    print(f"Wrote {output_dir / 'data_audit.json'}")
    print(f"Wrote {output_dir / 'data_audit.md'}")


if __name__ == "__main__":
    main()
