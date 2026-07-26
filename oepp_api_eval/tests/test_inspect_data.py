import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_data import _window_count


class WindowCountTests(unittest.TestCase):
    def test_short_sequences_contribute_one_left_padded_window(self) -> None:
        records = [
            {"anno": [object()]},
            {"anno": [object(), object()]},
            {"anno": [object(), object(), object()]},
            {"anno": [object(), object(), object(), object()]},
            {"anno": [object(), object(), object(), object(), object()]},
        ]

        self.assertEqual(_window_count(records, 3), 8)
        self.assertEqual(_window_count(records, 4), 6)

    def test_empty_sequences_do_not_invent_a_window(self) -> None:
        self.assertEqual(_window_count([{"anno": []}], 4), 0)


if __name__ == "__main__":
    unittest.main()
