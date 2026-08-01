from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from oepp.evaluation.planning import summarize_files, summarize_planning_rows


def _row(
    sample_id: str,
    dataset: str,
    task_id: str,
    step_index: int,
    gt_action: str,
    predicted_action: str,
    correct: int,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "task_id": task_id,
        "step_index": str(step_index),
        "gt_action": gt_action,
        "predicted_action": predicted_action,
        "correct": str(correct),
    }


class PlanningMetricTests(unittest.TestCase):
    def test_reports_distinct_window_micro_and_event_macro_metrics(self) -> None:
        rows = [
            _row("a-1", "COIN", "event-a", 0, "cut", "cut", 1),
            _row("a-1", "COIN", "event-a", 1, "mix", "mix", 1),
            _row("a-2", "COIN", "event-a", 0, "cut", "cut", 1),
            _row("a-2", "COIN", "event-a", 1, "mix", "mix", 1),
            _row("b-1", "CrossTask", "event-b", 0, "cut", "wash", 0),
            _row("b-1", "CrossTask", "event-b", 1, "mix", "wash", 0),
        ]
        summary = summarize_planning_rows(rows)
        self.assertAlmostEqual(summary["window_micro"]["sr"], 2 / 3)
        self.assertAlmostEqual(summary["window_micro"]["acc"], 2 / 3)
        self.assertAlmostEqual(summary["window_micro"]["miou"], 2 / 3)
        self.assertEqual(summary["event_macro"]["events"], 2)
        self.assertAlmostEqual(summary["event_macro"]["sr"], 0.5)
        self.assertAlmostEqual(summary["event_macro"]["acc"], 0.5)
        self.assertAlmostEqual(summary["event_macro"]["miou"], 0.5)

    def test_rejects_noncontiguous_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            summarize_planning_rows(
                [
                    _row("window", "COIN", "event", 0, "cut", "cut", 1),
                    _row("window", "COIN", "event", 2, "mix", "mix", 1),
                ]
            )

    def test_summarizes_base_and_novel_csvs_with_hashes(self) -> None:
        fields = [
            "sample_id",
            "dataset",
            "task_id",
            "step_index",
            "gt_action",
            "predicted_action",
            "correct",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {"base": root / "base.csv", "novel": root / "novel.csv"}
            for path in paths.values():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(_row(path.stem, "COIN", "event", 0, "cut", "cut", 1))
            report = summarize_files(paths)
        self.assertEqual(report["schema"], "oepp-planning-event-metrics-v1")
        self.assertEqual(set(report["splits"]), {"base", "novel"})
        self.assertEqual(report["splits"]["base"]["event_macro"]["events"], 1)


if __name__ == "__main__":
    unittest.main()
