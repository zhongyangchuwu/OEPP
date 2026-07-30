import os
import unittest
from pathlib import Path
from unittest.mock import patch

from oepp.data import FeatureKind, FeatureRoots, feature_dimension, resolve_feature_path
from oepp.data.features import VIDEOCLIP_ROOT_ENV, default_videoclip_root, repository_root


class FeaturePathTests(unittest.TestCase):
    def test_repository_default_is_machine_independent(self) -> None:
        with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: ""}, clear=False):
            self.assertEqual(
                default_videoclip_root(), repository_root() / "features" / "OEPP_videoclip"
            )

    def test_environment_override_controls_videoclip_root(self) -> None:
        configured_root = Path("/mnt/feature-store/videoclip")
        with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: str(configured_root)}):
            self.assertEqual(default_videoclip_root(), configured_root)

    def test_resolver_uses_feature_specific_canonical_paths(self) -> None:
        record = {
            "dataset": "COIN",
            "vid": "sample",
            "task_name": "Example",
            "task_id_old": 42,
        }
        roots = FeatureRoots(
            videoclip=Path("/features/videoclip"),
            coin_s3d=Path("/features/coin-s3d"),
            crosstask_s3d=Path("/features/crosstask-s3d"),
        )
        self.assertEqual(
            resolve_feature_path(record, FeatureKind.VIDEOCLIP, roots),
            Path("/features/videoclip/COIN_sample.npy"),
        )
        self.assertEqual(
            resolve_feature_path(record, FeatureKind.S3D, roots),
            Path("/features/coin-s3d/Example_42_sample.npy"),
        )

    def test_feature_dimensions_are_protocol_specific(self) -> None:
        self.assertEqual(feature_dimension("videoclip"), 768)
        self.assertEqual(feature_dimension("s3d"), 512)


if __name__ == "__main__":
    unittest.main()
