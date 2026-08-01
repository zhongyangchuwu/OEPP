"""Regression coverage for the standalone embedding preflight command."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class EmbeddingPreflightCliTests(unittest.TestCase):
    def test_help_runs_without_a_top_level_feature_paths_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "embedding_preflight.py", "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verify OEPP annotations", result.stdout)


if __name__ == "__main__":
    unittest.main()
