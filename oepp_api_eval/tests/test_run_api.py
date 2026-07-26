import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_api import _validate_manifest


class ExpectedSplitTests(unittest.TestCase):
    def test_rejects_manifest_for_other_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected base split"):
            _validate_manifest([{"sample_id": "novel-sample", "split": "novel"}], "base")

    def test_accepts_manifest_for_expected_split(self) -> None:
        _validate_manifest([{"sample_id": "base-sample", "split": "base"}], "base")


if __name__ == "__main__":
    unittest.main()
