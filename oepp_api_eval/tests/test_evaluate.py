import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import evaluate


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = [
            {
                "sample_id": "sample-1",
                "vid": "video-1",
                "event": "event-a",
                "split": "base",
                "T": 3,
                "image_setting": "1+1",
                "pool_type": "split",
                "candidate_order_seed": 42,
                "candidate_actions": [
                    {"id": 0, "text": "first"},
                    {"id": 1, "text": "second"},
                    {"id": 2, "text": "third"},
                ],
                "gt_actions": ["first", "second", "third"],
            },
            {
                "sample_id": "sample-2",
                "vid": "video-2",
                "event": "event-b",
                "split": "novel",
                "T": 3,
                "image_setting": "1+1",
                "pool_type": "split",
                "candidate_order_seed": 42,
                "candidate_actions": [
                    {"id": 0, "text": "first"},
                    {"id": 1, "text": "second"},
                    {"id": 2, "text": "third"},
                ],
                "gt_actions": ["first", "second", "third"],
            },
        ]

    def test_missing_prediction_remains_in_denominator(self) -> None:
        rows, report = evaluate(
            self.manifest,
            [{"sample_id": "sample-1", "parse_status": "ok", "action_ids": [0, 1, 2]}],
        )
        self.assertTrue(rows[0]["sr"])
        self.assertFalse(rows[1]["sr"])
        self.assertEqual(report["overall"]["samples"], 2)
        self.assertEqual(report["overall"]["SR"], 50.0)
        self.assertEqual(report["parse_or_missing_failures"], 1)

    def test_order_changes_accuracy_but_not_set_iou(self) -> None:
        rows, _ = evaluate(
            self.manifest[:1],
            [{"sample_id": "sample-1", "parse_status": "ok", "action_ids": [2, 1, 0]}],
        )
        self.assertFalse(rows[0]["sr"])
        self.assertEqual(rows[0]["acc"], 1 / 3)
        self.assertEqual(rows[0]["iou"], 100.0)


if __name__ == "__main__":
    unittest.main()
