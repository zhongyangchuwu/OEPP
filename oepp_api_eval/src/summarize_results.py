from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['samples']} | {metrics['SR']:.2f} | "
        f"{metrics['Acc']:.2f} | {metrics['mIoU']:.2f} |"
    )


def render(metrics: dict[str, Any]) -> str:
    scoring_mode = metrics.get("scoring_mode", "strict")
    strict_failures = metrics["parse_or_missing_failures"]
    total_samples = metrics["overall"]["samples"]
    strict_failure_rate = metrics["parse_or_missing_failure_rate"]
    lines = [
        "# OEPP API Evaluation Summary",
        "",
        f"Generated: `{metrics['generated_at']}`",
        f"Scoring mode: `{scoring_mode}`",
        "",
        "## Overall",
        "",
        "| Scope | Samples | SR | Acc | mIoU |",
        "|---|---:|---:|---:|---:|",
        _row("Overall", metrics["overall"]),
        "",
        "## Split metrics",
        "",
        "| Scope | Samples | SR | Acc | mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(_row(name, values) for name, values in metrics["by_split"].items())
    lines.extend(
        [
            "",
            "## Event-level macro average",
            "",
            f"- SR: **{metrics['event_macro']['SR']:.2f}**",
            f"- Acc: **{metrics['event_macro']['Acc']:.2f}**",
            f"- mIoU: **{metrics['event_macro']['mIoU']:.2f}**",
            "",
            "## Failures",
            "",
            f"- Strict parser or missing-prediction failures: **{strict_failures}** / "
            f"**{total_samples}** ({strict_failure_rate:.2f}%).",
        ]
    )
    if scoring_mode == "paper_compatible":
        lines.extend(
            [
                f"- Structurally valid raw-action sequences scored by the paper-compatible "
                f"rule: **{metrics['paper_compatible_raw_action_sequences']}**.",
                "- Every sample remains in the denominator. Formatted out-of-pool action names "
                "retain position-wise/set credit under the historical Table V scoring rule.",
            ]
        )
    else:
        lines.append(
            "- Failures remain in every denominator and receive zero score; this summary does not "
            "repair, omit, or substitute model outputs."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a concise OEPP evaluation summary from metrics JSON."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.metrics.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render(metrics), encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
