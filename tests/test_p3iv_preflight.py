from __future__ import annotations

import unittest
from pathlib import Path

from oepp.baselines.p3iv import (
    UPSTREAM_COIN_INPUT_DIMENSION,
    UPSTREAM_CROSSTASK_INPUT_DIMENSION,
    inspect_p3iv_compatibility,
)
from oepp.data import SplitBundle


class P3IVPreflightTests(unittest.TestCase):
    def test_split_one_preflight_blocks_unmodified_p3iv(self) -> None:
        bundle = SplitBundle.load(Path("data"), "split-001")
        report = inspect_p3iv_compatibility(
            bundle, "videoclip", Path("../upstreams/procedure-planning")
        )
        profiles = {profile.dataset: profile for profile in report.oepp_dataset_profiles}

        self.assertEqual(report.status, "blocked")
        self.assertEqual(report.selected_feature_dimension, 768)
        self.assertEqual(UPSTREAM_COIN_INPUT_DIMENSION, 512)
        self.assertEqual(UPSTREAM_CROSSTASK_INPUT_DIMENSION, 640)
        self.assertEqual(report.base_class_count, 122)
        self.assertEqual(report.novel_class_count, 55)
        self.assertEqual(report.total_class_count, 161)
        self.assertEqual(len(report.novel_actions_outside_train_class_space), 39)
        self.assertEqual(len(report.base_novel_pool_overlap), 16)
        self.assertIn("COIN", profiles)
        self.assertIn("CrossTask", profiles)
        self.assertFalse(report.upstream_source_paths["datasets"])
        self.assertFalse(report.upstream_source_paths["models"])
        self.assertFalse(report.as_json()["training_permitted"])
        self.assertTrue(any("VGGish" in blocker for blocker in report.blockers))
        self.assertTrue(any("license" in blocker.casefold() for blocker in report.blockers))


if __name__ == "__main__":
    unittest.main()
