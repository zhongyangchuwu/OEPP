import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_tablev_manifest import build_manifest
from extract_tablev_frames import PROTOCOL_ID, extract
from parse_response import LEGACY_NUMBERED_ACTION_NAMES
from run_api import TABLE_V_LEGACY_PROMPT_MODE, _prepare_request


class TableVManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = ["first action", "second action", "third action", "fourth action"]
        self.record = {
            "dataset": "COIN",
            "vid": "video-1",
            "task_name": "Example event",
            "anno": [
                {"action": action, "segment": [index * 10.0, index * 10.0 + 9.0]}
                for index, action in enumerate(self.actions)
            ],
        }

    def _observation(self) -> dict[str, object]:
        return {
            "sample_id": "tablev_T4_base_0000_video-1",
            "split": "base",
            "sequence_index": 0,
            "vid": "video-1",
            "source_video_path": "/server/videos/video-1.mp4",
            "start_f": 0.0,
            "end_f": 39.0,
            "action_list": self.actions,
            "image_setting": "3+3",
            "protocol": PROTOCOL_ID,
            "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [37.0, 38.0, 39.0]},
            "start_images": ["base/start_1.jpg", "base/start_2.jpg", "base/start_3.jpg"],
            "end_images": ["base/goal_1.jpg", "base/goal_2.jpg", "base/goal_3.jpg"],
        }

    def test_builds_portable_manifest_from_relative_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame_root = Path(temporary)
            observation = self._observation()
            for relative_path in [*observation["start_images"], *observation["end_images"]]:
                path = frame_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color="blue").save(path)
            manifest = build_manifest(
                [observation], [self.record], self.actions, frame_root, "base"
            )
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["event"], "Example event")
        self.assertEqual(manifest[0]["gt_action_ids"], [0, 1, 2, 3])
        self.assertEqual(manifest[0]["response_parser"], LEGACY_NUMBERED_ACTION_NAMES)

    def test_records_missing_source_video_without_importing_opencv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_file = root / "sequences.json"
            sequence_file.write_text(
                json.dumps(
                    [
                        {
                            "vid": "missing-video",
                            "video_path": str(root / "missing.mp4"),
                            "start_f": 0.0,
                            "end_f": 5.0,
                            "action_list": self.actions,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            observations, unavailable = extract(sequence_file, "base", root / "frames", None)
        self.assertEqual(observations, [])
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["error_type"], "FileNotFoundError")


class TableVRequestTests(unittest.TestCase):
    def test_places_three_start_images_before_three_goal_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_paths = []
            for index in range(6):
                path = root / f"image-{index}.jpg"
                Image.new("RGB", (8, 8), color="green").save(path)
                image_paths.append(str(path))
            sample = {
                "sample_id": "sample-1",
                "T": 4,
                "image_setting": "3+3",
                "start_images": image_paths[:3],
                "end_images": image_paths[3:],
                "candidate_actions": [
                    {"id": 0, "text": "first action"},
                    {"id": 1, "text": "second action"},
                ],
                "candidate_order_seed": None,
                "pool_type": "split",
            }
            template = json.dumps(
                {
                    "system_messages": ["Choose from: {candidate_action_list}"],
                    "user_start": "T is {T}; start.",
                    "user_end": "goal.",
                }
            )
            messages, metadata = _prepare_request(
                sample, template, TABLE_V_LEGACY_PROMPT_MODE, LEGACY_NUMBERED_ACTION_NAMES
            )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        user_content = messages[-1]["content"]
        image_positions = [
            index for index, item in enumerate(user_content) if item["type"] == "image_url"
        ]
        self.assertEqual(len(image_positions), 6)
        self.assertLess(
            image_positions[2],
            next(index for index, item in enumerate(user_content) if item.get("text") == "goal."),
        )
        self.assertEqual(metadata["response_parser"], LEGACY_NUMBERED_ACTION_NAMES)


if __name__ == "__main__":
    unittest.main()
