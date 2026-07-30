from __future__ import annotations

import unittest
from pathlib import Path

from oepp.baselines.kepp import build_train_only_pkg, inspect_kepp_compatibility
from oepp.data import SplitBundle


class KEPPPreflightTests(unittest.TestCase):
    def test_train_only_pkg_contains_only_edges_from_supplied_train_records(self) -> None:
        train_records = (
            {"anno": [{"action": "a"}, {"action": "b"}, {"action": "c"}]},
            {"anno": [{"action": "b"}, {"action": "c"}]},
        )

        pkg = build_train_only_pkg(train_records, ("a", "b", "c"))

        self.assertEqual(pkg.source_record_count, 2)
        self.assertEqual(pkg.action_to_id, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(pkg.edge_counts, {("a", "b"): 1, ("b", "c"): 2})
        self.assertNotIn(("c", "novel-only"), pkg.edge_counts)

    def test_split_one_preflight_blocks_unapproved_closed_set_port(self) -> None:
        bundle = SplitBundle.load(Path("data"), "split-001")

        report = inspect_kepp_compatibility(bundle, "videoclip")
        partitions = {item.partition: item for item in report.partitions}

        self.assertEqual(report.status, "blocked")
        self.assertEqual(report.train_class_count, 122)
        self.assertEqual(report.selected_observation_dimension, 2304)
        self.assertTrue(report.requires_feature_adapter)
        self.assertEqual(len(partitions["novel_test"].actions_outside_train_class_space), 39)
        self.assertEqual(len(partitions["novel_test"].actions_unseen_in_train_records), 39)
        self.assertEqual(len(report.base_novel_pool_overlap), 16)
        self.assertFalse(report.as_json()["training_permitted"])
        summary = report.summary_json()
        self.assertEqual(summary["pkg"]["action_class_count"], 122)
        self.assertGreater(summary["pkg"]["edge_count"], 0)
        self.assertNotIn("edge_counts", summary["pkg"])
        self.assertIn("edge_counts", report.as_json()["pkg"])
        self.assertEqual(report.pkg.source_record_count, len(bundle.partition_records("train")))


if __name__ == "__main__":
    unittest.main()
