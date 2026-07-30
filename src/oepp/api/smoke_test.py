from __future__ import annotations

import argparse
from pathlib import Path

from .common import API_CONFIG_ROOT, load_jsonl
from .run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exactly one approved OEPP Qwen3-VL smoke-test sample."
    )
    parser.add_argument("--config", type=Path, default=API_CONFIG_ROOT / "qwen3vl.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    manifest = load_jsonl(arguments.manifest)
    if len(manifest) != 1:
        parser.error("Smoke-test manifest must contain exactly one fully validated sample.")
    result = run(arguments.config, arguments.manifest, arguments.run_name, requested_limit=1)
    if result["api_attempts"] != 1 or result["completed_samples"] != 1:
        raise SystemExit(f"Smoke test did not complete one API request: {result}")
    print("Smoke test completed one request and saved raw artifacts.")


if __name__ == "__main__":
    main()
