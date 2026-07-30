from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oepp.api.replay_tablev import replay
from oepp.data import ActionPool, Partition


class ReplayBundle:
    split_id = "synthetic-split"
    source_hashes = {"records.base_test": "synthetic"}

    def __init__(self) -> None:
        self.records = (
            {
                "vid": "video-1",
                "task_name": "Synthetic event",
                "anno": [
                    {"action": "a", "segment": [0.0, 9.0]},
                    {"action": "b", "segment": [10.0, 19.0]},
                    {"action": "c", "segment": [20.0, 29.0]},
                    {"action": "a", "segment": [30.0, 39.0]},
                ],
            },
        )

    def partition_records(self, partition: Partition) -> tuple[dict[str, object], ...]:
        if partition is not Partition.BASE_TEST:
            raise AssertionError(f"Unexpected partition read: {partition}")
        return self.records

    def action_pool(self, pool: ActionPool) -> tuple[str, ...]:
        if pool is not ActionPool.BASE:
            raise AssertionError(f"Unexpected pool read: {pool}")
        return ("a", "b", "c")


class HistoricalTableVReplayTests(unittest.TestCase):
    def test_replay_preserves_raw_credit_but_blocks_out_of_pool_strict_output(self) -> None:
        legacy_results = [
            {
                "vid": "video-1",
                "start_f": 0.0,
                "end_f": 29.0,
                "action_list": ["a", "b", "c"],
                "output_list": ["a", "not in pool", "c"],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-results.json"
            path.write_text(json.dumps(legacy_results), encoding="utf-8")
            rows, report = replay(path, ReplayBundle(), "base", 3)

        self.assertEqual(
            report["coverage"],
            {
                "source_window_count": 2,
                "legacy_result_count": 1,
                "matched_window_count": 1,
                "missing_source_window_count": 1,
            },
        )
        self.assertEqual(report["strict"]["overall"]["Acc"], 0.0)
        self.assertEqual(report["strict"]["parse_or_missing_failures"], 1)
        self.assertAlmostEqual(report["paper_compatible"]["overall"]["Acc"], 200.0 / 3.0)
        self.assertEqual(
            rows[0]["strict"]["prediction_status"], "legacy_out_of_pool_or_ambiguous_output"
        )
        self.assertEqual(rows[0]["paper_compatible"]["list_pred"], ["a", "not in pool", "c"])


if __name__ == "__main__":
    unittest.main()
