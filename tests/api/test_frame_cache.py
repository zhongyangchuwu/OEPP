import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from oepp.api.build_tablev_manifest import build_manifest
from oepp.api.common import sha256_file
from oepp.api.compose_frame_cache import compose_observations
from oepp.api.extract_frame_cache import extract_cache_frames
from oepp.api.frame_cache import (
    VideoSource,
    build_cache_plan,
    build_video_source_index,
    write_cache_plan,
)
from oepp.api.tablev import TableVFrameProtocol
from oepp.api.verify_frame_cache_parity import verify_parity
from oepp.data import Partition


class _Bundle:
    split_id = "split-test"
    source_hashes = {"annotations": "abc"}

    def __init__(
        self, base_records: list[dict[str, object]], novel_records: list[dict[str, object]]
    ):
        self._records = {
            Partition.BASE_TEST: base_records,
            Partition.NOVEL_TEST: novel_records,
        }

    def partition_records(self, partition: Partition) -> list[dict[str, object]]:
        return self._records[partition]


def _record(dataset: str, vid: str, actions: list[str]) -> dict[str, object]:
    return {
        "dataset": dataset,
        "vid": vid,
        "task_name": f"{vid} event",
        "anno": [
            {"action": action, "segment": [float(index * 10), float(index * 10 + 9)]}
            for index, action in enumerate(actions)
        ],
    }


class SharedFrameCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_actions = ["base first", "base second", "base third"]
        self.novel_actions = ["novel first", "novel second", "novel third"]
        self.base_record = _record("COIN", "base-video", self.base_actions)
        self.novel_record = _record("CrossTask", "novel-video", self.novel_actions)
        self.bundle = _Bundle([self.base_record], [self.novel_record])
        self.sources = {
            ("COIN", "base-video"): VideoSource("COIN", "base-video", "/videos/base.mp4"),
            ("CrossTask", "novel-video"): VideoSource(
                "CrossTask", "novel-video", "/videos/novel.mp4"
            ),
        }

    def _all_protocols(self) -> list[TableVFrameProtocol]:
        return [
            TableVFrameProtocol(horizon, image_setting)
            for horizon in (3, 4)
            for image_setting in ("1+1", "3+3")
        ]

    def test_plans_all_horizon_and_image_setting_combinations_once(self) -> None:
        requests, observations, metadata = build_cache_plan(
            self.bundle, self.sources, self._all_protocols(), "tablev-shared-v2"
        )
        self.assertEqual(metadata["observation_count"], 8)
        self.assertEqual(len({request.frame_id for request in requests}), len(requests))
        self.assertEqual(metadata["frame_request_count"], len(requests))
        protocols = {observation["protocol"] for observation in observations}
        self.assertEqual(protocols, {protocol.identifier for protocol in self._all_protocols()})
        single_image = next(
            observation
            for observation in observations
            if observation["split"] == "base"
            and observation["protocol"] == TableVFrameProtocol(3, "1+1").identifier
        )
        self.assertEqual(len(single_image["frame_ids"]["start"]), 1)
        self.assertEqual(len(single_image["frame_ids"]["goal"]), 1)
        self.assertEqual(single_image["frame_timestamps"]["start"], [0.0])
        self.assertEqual(single_image["frame_timestamps"]["goal"], [29.0])

    def test_plans_padded_short_sequences(self) -> None:
        short_record = _record("COIN", "short-video", ["first", "second"])
        short_bundle = _Bundle([short_record], [self.novel_record])
        short_sources = {
            **self.sources,
            ("COIN", "short-video"): VideoSource("COIN", "short-video", "/videos/short.mp4"),
        }
        _, observations, _ = build_cache_plan(
            short_bundle,
            short_sources,
            [TableVFrameProtocol(3, "1+1")],
            "tablev-shared-v2",
        )
        base_observation = next(
            observation for observation in observations if observation["split"] == "base"
        )
        self.assertEqual(base_observation["action_list"], ["first", "first", "second"])
        self.assertEqual(base_observation["end_f"], 19.0)

    def test_builds_source_index_from_frozen_window_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_sequences = root / "base.json"
            novel_sequences = root / "novel.json"
            base_sequences.write_text(
                json.dumps(
                    [
                        {
                            "vid": "base-video",
                            "video_path": "/videos/base.mp4",
                            "start_f": 0.0,
                            "end_f": 29.0,
                            "action_list": self.base_actions,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            novel_sequences.write_text(
                json.dumps(
                    [
                        {
                            "vid": "novel-video",
                            "video_path": "/videos/novel.mp4",
                            "start_f": 0.0,
                            "end_f": 29.0,
                            "action_list": self.novel_actions,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            index = build_video_source_index(
                self.bundle, {"base": base_sequences, "novel": novel_sequences}
            )
        self.assertEqual(index["split_id"], "split-test")
        self.assertEqual(
            index["sources"],
            [
                {"dataset": "COIN", "vid": "base-video", "video_path": "/videos/base.mp4"},
                {
                    "dataset": "CrossTask",
                    "vid": "novel-video",
                    "video_path": "/videos/novel.mp4",
                },
            ],
        )

    def test_composes_shared_frames_and_builds_single_image_manifest(self) -> None:
        protocol = TableVFrameProtocol(3, "1+1")
        requests, observations, _ = build_cache_plan(
            self.bundle, self.sources, [protocol], "tablev-shared-v2"
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            outcomes: list[dict[str, object]] = []
            for request in requests:
                path = cache_root / request.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color="blue").save(path)
                outcomes.append(
                    {
                        "frame_id": request.frame_id,
                        "relative_path": request.relative_path,
                        "status": "available",
                        "image_sha256": sha256_file(path),
                    }
                )
            available, unavailable = compose_observations(
                observations, outcomes, cache_root, protocol, "base"
            )
            manifest = build_manifest(
                available,
                [self.base_record],
                self.base_actions,
                cache_root,
                "base",
                horizon=3,
                image_setting="1+1",
                expected_protocol=protocol.identifier,
            )
        self.assertEqual(unavailable, [])
        self.assertEqual(len(available), 1)
        self.assertEqual(len(available[0]["start_images"]), 1)
        self.assertEqual(len(available[0]["end_images"]), 1)
        self.assertEqual(manifest[0]["protocol"], protocol.identifier)
        self.assertEqual(manifest[0]["frame_cache_id"], "tablev-shared-v2")
        self.assertEqual(len(manifest[0]["start_images"]), 1)
        self.assertEqual(len(manifest[0]["end_images"]), 1)

    def test_preserves_unavailable_frame_dependencies(self) -> None:
        protocol = TableVFrameProtocol(3, "1+1")
        _, observations, _ = build_cache_plan(
            self.bundle, self.sources, [protocol], "tablev-shared-v2"
        )
        outcome_ids = observations[0]["frame_ids"]
        outcomes = [
            {"frame_id": frame_id, "status": "unavailable_frame", "error": "missing source"}
            for frame_id in [*outcome_ids["start"], *outcome_ids["goal"]]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            available, unavailable = compose_observations(
                observations, outcomes, Path(temporary), protocol, "base"
            )
        self.assertEqual(available, [])
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["status"], "unavailable_observation")

    def test_records_missing_videos_without_importing_opencv(self) -> None:
        request = {
            "frame_id": "missing-frame",
            "dataset": "COIN",
            "vid": "missing-video",
            "video_path": "/not-a-video/missing.mp4",
            "requested_timestamp": 0.0,
            "relative_path": "frames/missing-frame.jpg",
        }
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = extract_cache_frames([request], Path(temporary))
        self.assertEqual(outcomes[0]["status"], "unavailable_frame")
        self.assertEqual(outcomes[0]["error_type"], "FileNotFoundError")

    def test_seeds_exact_jpeg_bytes_from_legacy_observations(self) -> None:
        request = {
            "frame_id": "seeded-frame",
            "dataset": "COIN",
            "vid": "seeded-video",
            "video_path": "/not-a-video/seeded-video.mp4",
            "requested_timestamp": 0.0,
            "relative_path": "frames/seeded-frame.jpg",
        }
        legacy = {
            "sample_id": "legacy-sample",
            "protocol": "table_v_t4_3x3_legacy_v1",
            "image_setting": "3+3",
            "source_video_path": request["video_path"],
            "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [3.0, 4.0, 5.0]},
            "start_images": ["start-0.jpg", "start-1.jpg", "start-2.jpg"],
            "end_images": ["goal-0.jpg", "goal-1.jpg", "goal-2.jpg"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            for index, color in enumerate(["red", "green", "blue", "purple", "orange", "yellow"]):
                role = "start" if index < 3 else "goal"
                path = legacy_root / f"{role}-{index % 3}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=color).save(path)
            outcomes = extract_cache_frames(
                [request],
                root / "cache",
                legacy_observations=[legacy],
                legacy_frame_root=legacy_root,
            )
            seeded_path = root / "cache" / request["relative_path"]
            source_path = legacy_root / "start-0.jpg"
            self.assertEqual(seeded_path.read_bytes(), source_path.read_bytes())
            self.assertEqual(outcomes[0]["status"], "available")
            self.assertFalse(outcomes[0]["reused"])
            self.assertEqual(outcomes[0]["image_sha256"], sha256_file(source_path))
            self.assertEqual(outcomes[0]["provenance"]["kind"], "legacy_t4_3x3_frame")

    def test_rejects_conflicting_legacy_jpegs_for_one_frame(self) -> None:
        request = {
            "frame_id": "conflicted-frame",
            "dataset": "COIN",
            "vid": "conflicted-video",
            "video_path": "/not-a-video/conflicted-video.mp4",
            "requested_timestamp": 0.0,
            "relative_path": "frames/conflicted-frame.jpg",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            observations = []
            for index, color in enumerate(["red", "blue"]):
                prefix = f"duplicate-{index}"
                for position in range(6):
                    path = legacy_root / f"{prefix}-{position}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=color).save(path)
                observations.append(
                    {
                        "sample_id": prefix,
                        "protocol": "table_v_t4_3x3_legacy_v1",
                        "image_setting": "3+3",
                        "source_video_path": request["video_path"],
                        "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [3.0, 4.0, 5.0]},
                        "start_images": [f"{prefix}-0.jpg", f"{prefix}-1.jpg", f"{prefix}-2.jpg"],
                        "end_images": [f"{prefix}-3.jpg", f"{prefix}-4.jpg", f"{prefix}-5.jpg"],
                    }
                )
            with self.assertRaisesRegex(ValueError, "legacy observations disagree"):
                extract_cache_frames(
                    [request],
                    root / "cache",
                    legacy_observations=observations,
                    legacy_frame_root=legacy_root,
                )

    def test_selects_a_deterministic_source_for_identical_legacy_jpegs(self) -> None:
        request = {
            "frame_id": "deterministic-frame",
            "dataset": "COIN",
            "vid": "deterministic-video",
            "video_path": "/not-a-video/deterministic-video.mp4",
            "requested_timestamp": 0.0,
            "relative_path": "frames/deterministic-frame.jpg",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            observations = []
            for prefix in ("zeta", "alpha"):
                for position in range(6):
                    path = legacy_root / f"{prefix}-{position}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color="blue").save(path)
                observations.append(
                    {
                        "sample_id": prefix,
                        "protocol": "table_v_t4_3x3_legacy_v1",
                        "image_setting": "3+3",
                        "source_video_path": request["video_path"],
                        "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [3.0, 4.0, 5.0]},
                        "start_images": [f"{prefix}-0.jpg", f"{prefix}-1.jpg", f"{prefix}-2.jpg"],
                        "end_images": [f"{prefix}-3.jpg", f"{prefix}-4.jpg", f"{prefix}-5.jpg"],
                    }
                )
            outcomes = extract_cache_frames(
                [request],
                root / "cache",
                legacy_observations=observations,
                legacy_frame_root=legacy_root,
            )
            self.assertEqual(
                outcomes[0]["provenance"]["source_path"], str(legacy_root / "alpha-0.jpg")
            )

    def test_rejects_a_non_jpeg_legacy_frame(self) -> None:
        request = {
            "frame_id": "non-jpeg-frame",
            "dataset": "COIN",
            "vid": "non-jpeg-video",
            "video_path": "/not-a-video/non-jpeg-video.mp4",
            "requested_timestamp": 0.0,
            "relative_path": "frames/non-jpeg-frame.jpg",
        }
        legacy = {
            "sample_id": "legacy-non-jpeg",
            "protocol": "table_v_t4_3x3_legacy_v1",
            "image_setting": "3+3",
            "source_video_path": request["video_path"],
            "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [3.0, 4.0, 5.0]},
            "start_images": ["start-0.jpg", "start-1.jpg", "start-2.jpg"],
            "end_images": ["goal-0.jpg", "goal-1.jpg", "goal-2.jpg"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            for index, raw_path in enumerate([*legacy["start_images"], *legacy["end_images"]]):
                path = legacy_root / raw_path
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color="blue").save(
                    path, format="PNG" if index == 0 else "JPEG"
                )
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                extract_cache_frames(
                    [request],
                    root / "cache",
                    legacy_observations=[legacy],
                    legacy_frame_root=legacy_root,
                )

    def test_verifies_t4_three_image_cache_parity(self) -> None:
        colors = ["red", "green", "blue", "purple", "orange", "yellow"]
        protocol = TableVFrameProtocol(4, "3+3")
        legacy = {
            "sample_id": "legacy-sample",
            "protocol": "table_v_t4_3x3_legacy_v1",
            "image_setting": "3+3",
            "vid": "base-video",
            "start_f": 0.0,
            "end_f": 39.0,
            "action_list": ["one", "two", "three", "four"],
            "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [37.0, 38.0, 39.0]},
            "start_images": ["start-0.jpg", "start-1.jpg", "start-2.jpg"],
            "end_images": ["goal-0.jpg", "goal-1.jpg", "goal-2.jpg"],
        }
        shared = {
            **legacy,
            "sample_id": "shared-sample",
            "protocol": protocol.identifier,
            "frame_timestamps": {"start": [0.0, 1.0, 2.0], "goal": [37.0, 38.0, 39.0]},
            "start_images": ["frames/start-0.jpg", "frames/start-1.jpg", "frames/start-2.jpg"],
            "end_images": ["frames/goal-0.jpg", "frames/goal-1.jpg", "frames/goal-2.jpg"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_root = root / "legacy"
            shared_root = root / "shared"
            for index, color in enumerate(colors):
                role = "start" if index < 3 else "goal"
                position = index if index < 3 else index - 3
                for image_root, relative_path in (
                    (legacy_root, f"{role}-{position}.jpg"),
                    (shared_root, f"frames/{role}-{position}.jpg"),
                ):
                    path = image_root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=color).save(path)
            report = verify_parity([legacy], legacy_root, [shared], shared_root)
            shared["frame_timestamps"]["goal"][0] = 36.0
            mismatch_report = verify_parity([legacy], legacy_root, [shared], shared_root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["mismatch_count"], 0)
        self.assertFalse(mismatch_report["passed"])
        self.assertEqual(mismatch_report["mismatch_count"], 1)

    def test_writes_a_new_isolated_plan_directory(self) -> None:
        requests, observations, metadata = build_cache_plan(
            self.bundle, self.sources, self._all_protocols(), "tablev-shared-v2"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            write_cache_plan(output, requests, observations, metadata)
            self.assertTrue((output / "frame_requests.jsonl").is_file())
            self.assertTrue((output / "observations.jsonl").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            with self.assertRaises(FileExistsError):
                write_cache_plan(output, requests, observations, metadata)


if __name__ == "__main__":
    unittest.main()
