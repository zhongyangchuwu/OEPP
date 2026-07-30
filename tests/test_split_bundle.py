from __future__ import annotations

import unittest
from pathlib import Path

from oepp.data import (
    FeatureKind,
    PaddingPolicy,
    Partition,
    SequenceDataset,
    SplitBundle,
    build_windows,
)


class SplitBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = SplitBundle.load(Path("data"), "split-001")

    def test_canonical_split_records_and_pools_are_validated(self) -> None:
        self.assertEqual(len(self.bundle.partition_records(Partition.TRAIN)), 1285)
        self.assertEqual(len(self.bundle.partition_records(Partition.VALIDATION)), 337)
        self.assertEqual(len(self.bundle.partition_records(Partition.BASE_TEST)), 416)
        self.assertEqual(len(self.bundle.partition_records(Partition.NOVEL_TEST)), 733)
        self.assertEqual(len(self.bundle.action_pool("base")), 122)
        self.assertEqual(len(self.bundle.action_pool("novel")), 55)
        self.assertEqual(len(self.bundle.action_pool("total")), 161)
        self.assertEqual(self.bundle.provenance["kind"], "canonical_paper_split")
        self.assertEqual(len(self.bundle.source_hashes), 10)

    def test_sequence_dataset_uses_the_bundle_and_feature_contract(self) -> None:
        dataset = SequenceDataset(
            self.bundle,
            Partition.TRAIN,
            feature=FeatureKind.VIDEOCLIP,
            horizon=3,
            padding=PaddingPolicy.LEFT,
        )
        self.assertEqual(len(dataset), 3550)
        sample = dataset[0]
        self.assertEqual(sample[1].shape, (3 * 768,))
        self.assertEqual(sample[2].shape, (3 * 768,))
        self.assertEqual(sample[3].shape, (6 * 768,))
        self.assertEqual(sample[5].shape, (3, 768))
        self.assertEqual(sample[7].shape, (3,))
        self.assertEqual(dataset.metadata_at(0)["split_id"], "split-001")

    def test_left_padding_contributes_one_stable_window(self) -> None:
        records = [
            {
                "dataset": "COIN",
                "vid": "sample",
                "task_name": "Example",
                "task_id": 1,
                "task_id_old": 2,
                "anno": [
                    {"action": "first", "segment": [0, 1]},
                    {"action": "second", "segment": [2, 3]},
                ],
            }
        ]
        windows = build_windows(
            records,
            split_id="split-test",
            partition="train",
            horizon=3,
            padding=PaddingPolicy.LEFT,
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].sample_id, "split-test_train_COIN_sample_padded_T3")
        self.assertEqual(windows[0].actions, ("first", "first", "second"))
        self.assertTrue(windows[0].is_padded)
        self.assertEqual(windows[0].pad_count, 1)

    def test_empty_record_never_invents_a_window(self) -> None:
        self.assertEqual(
            build_windows(
                [{"dataset": "COIN", "vid": "empty", "anno": []}],
                split_id="split-test",
                partition="train",
                horizon=3,
                padding=PaddingPolicy.LEFT,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
