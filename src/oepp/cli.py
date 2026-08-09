"""Single command dispatcher for OEPP package entry points."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .data import Partition, SplitBundle

COMMAND_MODULES = {
    "train": "oepp.training.direct",
    "train-pdpp": "oepp.training.pdpp",
    "export": "oepp.exports.direct",
    "export-pdpp": "oepp.exports.pdpp",
    "kepp-preflight": "oepp.baselines.kepp",
    "p3iv-preflight": "oepp.baselines.p3iv",
    "import-alternate-split": "oepp.data.alternate_split",
    "derive-alternate-split": "oepp.data.derive_alternate_split",
    "derive-cluster-pairs": "oepp.data.cluster_pair",
    "summarize-planning-metrics": "oepp.evaluation.planning",
    "summarize-cluster-pairs": "oepp.evaluation.cluster_pair",
    "run-cluster-pairs": "oepp.experiments.cluster_pair",
    "api-build-manifest": "oepp.api.build_manifest",
    "api-build-tablev": "oepp.api.build_tablev_manifest",
    "api-extract-tablev": "oepp.api.extract_tablev_frames",
    "api-build-video-index": "oepp.api.build_video_source_index",
    "api-plan-frame-cache": "oepp.api.plan_frame_cache",
    "api-extract-frame-cache": "oepp.api.extract_frame_cache",
    "api-compose-frame-cache": "oepp.api.compose_frame_cache",
    "api-verify-frame-cache-parity": "oepp.api.verify_frame_cache_parity",
    "api-inspect": "oepp.api.inspect_data",
    "api-run": "oepp.api.run",
    "api-evaluate": "oepp.api.evaluate",
    "api-summary": "oepp.api.summarize_results",
    "api-replay-tablev": "oepp.api.replay_tablev",
    "api-smoke": "oepp.api.smoke_test",
}


def _split_audit(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize one OEPP split bundle")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-id", default="split-001")
    arguments = parser.parse_args(argv)
    bundle = SplitBundle.load(arguments.data_root, arguments.split_id)
    payload = {
        "split_id": bundle.split_id,
        "partitions": {
            partition.value: len(bundle.partition_records(partition)) for partition in Partition
        },
        "pool_sizes": {name.value: len(bundle.action_pool(name)) for name in bundle.pools},
        "source_hashes": dict(bundle.source_hashes),
        "provenance": dict(bundle.provenance),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = ", ".join(["split-audit", *COMMAND_MODULES])
    if not arguments:
        raise SystemExit(f"usage: oepp <command> [options]\ncommands: {commands}")
    command, command_argv = arguments[0], arguments[1:]
    if command == "split-audit":
        return _split_audit(command_argv)
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        raise SystemExit(f"unknown oepp command {command!r}; commands: {commands}")
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        raise RuntimeError(f"OEPP command module has no main(): {module_name}")
    original_argv = sys.argv
    try:
        sys.argv = [f"oepp {command}", *command_argv]
        entrypoint()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
