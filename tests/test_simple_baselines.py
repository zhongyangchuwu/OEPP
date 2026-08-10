from __future__ import annotations

import unittest

import torch

from oepp.baselines.simple import matching_action_indices, random_action_indices


class SimpleBaselineTests(unittest.TestCase):
    def test_random_is_sample_stable_and_without_replacement(self) -> None:
        arguments = {
            "pool_size": 11,
            "horizon": 4,
            "split_id": "split-002",
            "partition": "novel_test",
            "sample_id": "sample-17",
            "seed": 42,
        }
        first = random_action_indices(**arguments)
        second = random_action_indices(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(all(0 <= index < 11 for index in first))

    def test_random_rejects_pool_smaller_than_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least horizon"):
            random_action_indices(
                pool_size=2,
                horizon=3,
                split_id="split-001",
                partition="base_test",
                sample_id="sample",
            )

    def test_matching_decodes_endpoints_and_midpoint_intermediate(self) -> None:
        candidates = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        start = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        goal = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
        labels = matching_action_indices(start, goal, candidates, horizon=3)
        self.assertEqual(labels.tolist(), [[0, 2, 1]])

    def test_matching_uses_candidate_order_for_ties(self) -> None:
        candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        start = torch.tensor([[[1.0, 0.0]]])
        goal = torch.tensor([[[0.0, 1.0]]])
        labels = matching_action_indices(start, goal, candidates, horizon=3)
        self.assertEqual(labels.tolist(), [[0, 0, 2]])


if __name__ == "__main__":
    unittest.main()
