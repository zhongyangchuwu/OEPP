import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset.dataset import Video
from feature_paths import (
    DEFAULT_VIDEOCLIP_ROOT,
    VIDEOCLIP_ROOT_ENV,
    videoclip_feature_path,
    videoclip_root,
)


class VideoCLIPPathTests(unittest.TestCase):
    def test_repository_default_is_machine_independent(self) -> None:
        with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: ""}, clear=False):
            self.assertEqual(videoclip_root(), DEFAULT_VIDEOCLIP_ROOT)

    def test_environment_override_controls_dataset_and_preflight_path(self) -> None:
        configured_root = Path("/mnt/feature-store/videoclip")
        with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: str(configured_root)}):
            self.assertEqual(videoclip_root(), configured_root)
            self.assertEqual(
                videoclip_feature_path("COIN", "example-video"),
                configured_root / "COIN_example-video.npy",
            )

    def test_explicit_root_overrides_environment(self) -> None:
        explicit_root = Path("/tmp/explicit-videoclip-root")
        with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: "/ignored"}):
            self.assertEqual(videoclip_root(explicit_root), explicit_root)

    def test_video_loader_uses_configured_feature_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            data_root = temporary_root / "data"
            feature_root = temporary_root / "features"
            data_root.mkdir()
            feature_root.mkdir()
            (data_root / "train_train_base_dataset_1.json").write_text(
                json.dumps(
                    [
                        {
                            "dataset": "COIN",
                            "vid": "sample",
                            "anno": [{"segment": [0, 3], "action": "first"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            np.save(feature_root / "COIN_sample.npy", np.ones((4, 768), dtype=np.float32))

            with patch.dict(os.environ, {VIDEOCLIP_ROOT_ENV: str(feature_root)}):
                dataset = Video(str(data_root), split=1, feat="videoclip")
                vid, starts, ends, actions = dataset[0]

            self.assertEqual(vid, "sample")
            self.assertEqual(actions, ["first"])
            self.assertEqual(starts[0].shape[0], 3 * 768)
            self.assertEqual(ends[0].shape[0], 3 * 768)


if __name__ == "__main__":
    unittest.main()
