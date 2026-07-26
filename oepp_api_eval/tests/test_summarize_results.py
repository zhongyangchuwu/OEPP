import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summarize_results import render


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = {
            "generated_at": "2026-07-26T00:00:00+00:00",
            "overall": {"samples": 1, "SR": 0.0, "Acc": 50.0, "mIoU": 50.0},
            "by_split": {"base": {"samples": 1, "SR": 0.0, "Acc": 50.0, "mIoU": 50.0}},
            "event_macro": {"SR": 0.0, "Acc": 50.0, "mIoU": 50.0},
            "parse_or_missing_failures": 1,
            "parse_or_missing_failure_rate": 100.0,
            "paper_compatible_raw_action_sequences": 1,
        }

    def test_paper_compatible_summary_explains_raw_action_scoring(self) -> None:
        summary = render({**self.metrics, "scoring_mode": "paper_compatible"})
        self.assertIn("Scoring mode: `paper_compatible`", summary)
        self.assertIn("raw-action sequences", summary)
        self.assertIn("out-of-pool", summary)

    def test_strict_summary_explains_zero_score_failures(self) -> None:
        summary = render({**self.metrics, "scoring_mode": "strict"})
        self.assertIn("Scoring mode: `strict`", summary)
        self.assertIn("receive zero score", summary)


if __name__ == "__main__":
    unittest.main()
