"""Verify a new shared cache preserves the frozen Table V T=4, 3+3 visual contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import load_jsonl, sha256_file, write_json
from .tablev import TableVFrameProtocol, protocol_id, window_key


def _frame_path(frame_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str):
        raise ValueError("observation frame paths must be strings")
    root = frame_root.resolve()
    path = (root / raw_path).resolve()
    if root not in path.parents:
        raise ValueError(f"observation frame path escapes its frame root: {raw_path}")
    if not path.is_file():
        raise FileNotFoundError(f"observation frame does not exist: {path}")
    return path


def _observation_key(observation: Mapping[str, Any]) -> tuple[str, float, float, tuple[str, ...]]:
    vid = observation.get("vid")
    start = observation.get("start_f")
    end = observation.get("end_f")
    actions = observation.get("action_list")
    if (
        not isinstance(vid, str)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not isinstance(actions, list)
        or not all(isinstance(action, str) for action in actions)
    ):
        raise ValueError(f"{observation.get('sample_id')} has an invalid Table V identity")
    return window_key(vid, float(start), float(end), actions)


def _index_observations(
    observations: list[dict[str, Any]], expected_protocol: str
) -> dict[tuple[str, float, float, tuple[str, ...]], dict[str, Any]]:
    indexed: dict[tuple[str, float, float, tuple[str, ...]], dict[str, Any]] = {}
    for observation in observations:
        if observation.get("protocol") != expected_protocol:
            raise ValueError(
                f"{observation.get('sample_id')} does not use expected protocol {expected_protocol}"
            )
        if observation.get("image_setting") != "3+3":
            raise ValueError(f"{observation.get('sample_id')} is not a 3+3 observation")
        key = _observation_key(observation)
        if key in indexed:
            raise ValueError(f"duplicate observation identity in parity input: {key}")
        indexed[key] = observation
    return indexed


def _role_state(observation: Mapping[str, Any], frame_root: Path, role: str) -> dict[str, Any]:
    image_key = "start_images" if role == "start" else "end_images"
    timestamps = observation.get("frame_timestamps")
    if not isinstance(timestamps, Mapping):
        raise ValueError(f"{observation.get('sample_id')} lacks frame_timestamps")
    role_timestamps = timestamps.get(role)
    image_paths = observation.get(image_key)
    if (
        not isinstance(role_timestamps, list)
        or len(role_timestamps) != 3
        or not all(isinstance(value, (int, float)) for value in role_timestamps)
        or not isinstance(image_paths, list)
        or len(image_paths) != 3
    ):
        raise ValueError(f"{observation.get('sample_id')} has an invalid 3+3 {role} payload")
    paths = [_frame_path(frame_root, raw_path) for raw_path in image_paths]
    return {
        "timestamps": [float(value) for value in role_timestamps],
        "sha256": [sha256_file(path) for path in paths],
    }


def _identity_json(key: tuple[str, float, float, tuple[str, ...]]) -> dict[str, Any]:
    vid, start, end, actions = key
    return {"vid": vid, "start_f": start, "end_f": end, "action_list": list(actions)}


def verify_parity(
    legacy_observations: list[dict[str, Any]],
    legacy_frame_root: Path,
    shared_observations: list[dict[str, Any]],
    shared_frame_root: Path,
    max_mismatches: int = 20,
) -> dict[str, Any]:
    """Compare identities, timestamps, image order, and bytes for T=4 legacy 3+3 assets."""
    legacy = _index_observations(legacy_observations, protocol_id(4))
    shared_protocol = TableVFrameProtocol(4, "3+3").identifier
    shared = _index_observations(shared_observations, shared_protocol)
    mismatches: list[dict[str, Any]] = []
    all_keys = sorted(set(legacy).union(shared))
    for key in all_keys:
        legacy_observation = legacy.get(key)
        shared_observation = shared.get(key)
        if legacy_observation is None or shared_observation is None:
            mismatches.append(
                {
                    "identity": _identity_json(key),
                    "reason": "missing_observation",
                    "legacy_present": legacy_observation is not None,
                    "shared_present": shared_observation is not None,
                }
            )
        else:
            difference: dict[str, Any] = {}
            for role in ("start", "goal"):
                legacy_state = _role_state(legacy_observation, legacy_frame_root, role)
                shared_state = _role_state(shared_observation, shared_frame_root, role)
                if legacy_state != shared_state:
                    difference[role] = {"legacy": legacy_state, "shared": shared_state}
            if difference:
                mismatches.append(
                    {
                        "identity": _identity_json(key),
                        "reason": "frame_order_timestamp_or_hash_mismatch",
                        "legacy_sample_id": legacy_observation.get("sample_id"),
                        "shared_sample_id": shared_observation.get("sample_id"),
                        "difference": difference,
                    }
                )
        if len(mismatches) >= max_mismatches:
            break
    mismatch_count = sum(
        1 for key in all_keys if legacy.get(key) is None or shared.get(key) is None
    )
    if mismatch_count < len(all_keys):
        for key in all_keys:
            legacy_observation = legacy.get(key)
            shared_observation = shared.get(key)
            if legacy_observation is None or shared_observation is None:
                continue
            if any(
                _role_state(legacy_observation, legacy_frame_root, role)
                != _role_state(shared_observation, shared_frame_root, role)
                for role in ("start", "goal")
            ):
                mismatch_count += 1
    return {
        "legacy_protocol": protocol_id(4),
        "shared_protocol": shared_protocol,
        "legacy_observation_count": len(legacy),
        "shared_observation_count": len(shared),
        "paired_observation_count": len(set(legacy).intersection(shared)),
        "mismatch_count": mismatch_count,
        "passed": mismatch_count == 0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify T=4, 3+3 shared-cache parity with frozen legacy observations."
    )
    parser.add_argument("--legacy-observations", type=Path, required=True)
    parser.add_argument("--legacy-frame-root", type=Path, required=True)
    parser.add_argument("--shared-observations", type=Path, required=True)
    parser.add_argument("--shared-frame-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-mismatches", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {arguments.report}")
    if arguments.max_mismatches <= 0:
        parser.error("--max-mismatches must be positive")
    report = verify_parity(
        load_jsonl(arguments.legacy_observations),
        arguments.legacy_frame_root,
        load_jsonl(arguments.shared_observations),
        arguments.shared_frame_root,
        arguments.max_mismatches,
    )
    write_json(arguments.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "mismatches"}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
