import unittest

from oepp.api.evaluate import evaluate


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

    def test_paper_compatible_scoring_preserves_partial_out_of_pool_credit(self) -> None:
        prediction = {
            "sample_id": "sample-1",
            "parse_status": "parse_failure",
            "action_ids": None,
            "action_texts": [" First ", "outside candidate", "third"],
        }
        strict_rows, strict_report = evaluate(self.manifest[:1], [prediction])
        compatible_rows, compatible_report = evaluate(
            self.manifest[:1], [prediction], "paper_compatible"
        )
        self.assertEqual(strict_rows[0]["acc"], 0.0)
        self.assertEqual(strict_rows[0]["iou"], 0.0)
        self.assertAlmostEqual(compatible_rows[0]["acc"], 2 / 3)
        self.assertEqual(compatible_rows[0]["iou"], 50.0)
        self.assertTrue(compatible_rows[0]["scored_from_raw_action_texts"])
        self.assertEqual(compatible_report["paper_compatible_raw_action_sequences"], 1)
        self.assertEqual(strict_report["parse_or_missing_failures"], 1)


if __name__ == "__main__":
    unittest.main()
