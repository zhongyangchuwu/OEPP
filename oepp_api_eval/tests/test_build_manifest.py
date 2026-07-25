import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_manifest import _candidate_actions


class CandidateOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_pool = ["first", "second", "third", "fourth", "fifth"]

    def test_original_order_preserves_action_pool_order(self) -> None:
        candidates = _candidate_actions(self.action_pool, "original", None)
        self.assertEqual([candidate["id"] for candidate in candidates], list(range(5)))
        self.assertEqual([candidate["text"] for candidate in candidates], self.action_pool)

    def test_seeded_shuffle_is_reproducible(self) -> None:
        first = _candidate_actions(self.action_pool, "shuffled", 17)
        second = _candidate_actions(self.action_pool, "shuffled", 17)
        self.assertEqual(first, second)
        self.assertEqual(sorted(candidate["text"] for candidate in first), sorted(self.action_pool))

    def test_seeded_shuffle_requires_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a seed"):
            _candidate_actions(self.action_pool, "shuffled", None)


if __name__ == "__main__":
    unittest.main()
