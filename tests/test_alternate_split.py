from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from oepp.data import SplitBundle
from oepp.data.alternate_split import import_alternate_split
from oepp.data.derive_alternate_split import derive_membership


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _record(dataset: str, vid: str, task_id: int, action: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "vid": vid,
        "task_name": f"task-{task_id}",
        "task_id": task_id,
        "task_id_old": task_id,
        "anno": [{"action": action, "segment": [0.0, 1.0]}],
    }


class AlternateSplitImportTests(unittest.TestCase):
    def _source_data_root(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, list[dict[str, object]]]]:
        data_root = root / "data"
        records = {
            "train": [_record("COIN", "train-video", 1, "base action")],
            "validation": [_record("COIN", "validation-video", 1, "base action")],
            "base_test": [_record("COIN", "base-video", 1, "base action")],
            "novel_test": [_record("CrossTask", "novel-video", 2, "novel action")],
        }
        partitions = {
            name: {
                "path": f"../../{name}.json",
                "sha256": _write_json(data_root / f"{name}.json", value),
            }
            for name, value in records.items()
        }
        pools = {
            "base": ["base action"],
            "novel": ["novel action"],
            "total": ["base action", "novel action"],
        }
        action_pools = {
            name: {
                "path": f"../../{name}_pool.json",
                "sha256": _write_json(data_root / f"{name}_pool.json", value),
            }
            for name, value in pools.items()
        }
        task_info_path = data_root / "task_info.json"
        task_info = {"source": "synthetic"}
        action_embeddings = {}
        for feature, dimension in (("videoclip", 768), ("s3d", 512)):
            path = data_root / f"{feature}_actions.json"
            action_embeddings[feature] = {
                "path": f"../../{path.name}",
                "sha256": _write_json(
                    path,
                    {action: [0.0] * dimension for action in pools["total"]},
                ),
            }
        manifest = {
            "schema": "oepp-split-bundle-v1",
            "split_id": "split-source",
            "description": "synthetic source bundle",
            "provenance": {"kind": "test"},
            "partitions": partitions,
            "action_pools": action_pools,
            "task_info": {
                "path": "../../task_info.json",
                "sha256": _write_json(task_info_path, task_info),
            },
            "action_embeddings": action_embeddings,
        }
        _write_json(data_root / "splits" / "split-source" / "manifest.json", manifest)
        feature_root = root / "features"
        feature_root.mkdir()
        for rows in records.values():
            for row in rows:
                np.save(feature_root / f"{row['dataset']}_{row['vid']}.npy", np.zeros((2, 768)))
        return data_root, feature_root, records

    def _membership(
        self, source: SplitBundle, records: dict[str, list[dict[str, object]]]
    ) -> dict[str, object]:
        return {
            "schema": "oepp-alternate-split-membership-v1",
            "split_id": "split-alternate",
            "source_split": {
                "split_id": source.split_id,
                "source_hashes": dict(source.source_hashes),
            },
            "provenance": {
                "kind": "authoritative_event_split",
                "authority": "synthetic test authority",
                "source": "synthetic membership",
                "event_assignment_rule": "COIN task 1 is Base; CrossTask task 2 is Novel",
            },
            "partitions": {
                partition: [{"dataset": row["dataset"], "vid": row["vid"]} for row in rows]
                for partition, rows in records.items()
            },
        }

    def test_imports_full_authoritative_membership_as_valid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            membership_path = root / "membership.json"
            _write_json(membership_path, self._membership(source, records))
            report_path = root / "report.json"
            report = import_alternate_split(
                data_root, source, membership_path, feature_root, report_path
            )
            alternate = SplitBundle.load(data_root, "split-alternate")
            report_exists = report_path.is_file()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["coverage"], {"source_video_count": 4, "imported_video_count": 4})
        self.assertEqual(report["event_and_action_audit"]["base_novel_event_overlap"], [])
        self.assertEqual(report["event_and_action_audit"]["base_novel_action_overlap"], [])
        self.assertEqual(report["videoclip_feature_audit"]["checked"], 4)
        self.assertTrue(report_exists)
        self.assertEqual(alternate.split_id, "split-alternate")
        self.assertEqual(len(alternate.partition_records("novel_test")), 1)
        self.assertEqual(alternate.action_pool("total"), ("base action", "novel action"))

    def test_derives_and_imports_an_explicit_event_stratified_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            historical = root / "historical"
            sources = {
                "base_events": _write_json(historical / "base-events.json", ["task-1"]),
                "novel_events": _write_json(historical / "novel-events.json", ["task-2"]),
                "base_records": _write_json(
                    historical / "base-records.json",
                    [records["train"][0], records["validation"][0], records["base_test"][0]],
                ),
                "novel_records": _write_json(
                    historical / "novel-records.json", records["novel_test"]
                ),
                "base_train": _write_json(
                    historical / "base-train.json", [records["train"][0], records["validation"][0]]
                ),
                "base_test": _write_json(historical / "base-test.json", records["base_test"]),
            }
            membership_path = root / "split-derived-membership.json"
            membership = derive_membership(
                source=source,
                split_id="split-derived",
                base_events_path=historical / "base-events.json",
                novel_events_path=historical / "novel-events.json",
                base_records_path=historical / "base-records.json",
                novel_records_path=historical / "novel-records.json",
                base_train_path=historical / "base-train.json",
                base_test_path=historical / "base-test.json",
                validation_fraction=0.2,
                validation_seed=20260731,
                authority="OEPP authors",
                output=membership_path,
            )
            report = import_alternate_split(
                data_root, source, membership_path, feature_root, root / "derived-report.json"
            )
            description = json.loads(
                (data_root / "splits" / "split-derived" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )["description"]
        self.assertEqual(membership["provenance"]["kind"], "derived_event_split")
        self.assertEqual(membership["provenance"]["derivation"]["validation_seed"], 20260731)
        self.assertEqual(set(membership["provenance"]["derivation"]["source_files"]), set(sources))
        self.assertEqual(
            {name: len(rows) for name, rows in membership["partitions"].items()},
            {
                "train": 1,
                "validation": 1,
                "base_test": 1,
                "novel_test": 1,
            },
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            description,
            "Imported derived alternate OEPP event split; membership is copied verbatim.",
        )

    def test_rejects_historical_event_names_that_disagree_with_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, _, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            historical = root / "historical"
            _write_json(historical / "base-events.json", ["wrong-task"])
            _write_json(historical / "novel-events.json", ["task-2"])
            _write_json(
                historical / "base-records.json",
                [records["train"][0], records["validation"][0], records["base_test"][0]],
            )
            _write_json(historical / "novel-records.json", records["novel_test"])
            _write_json(
                historical / "base-train.json", [records["train"][0], records["validation"][0]]
            )
            _write_json(historical / "base-test.json", records["base_test"])
            with self.assertRaisesRegex(ValueError, "event-name lists"):
                derive_membership(
                    source=source,
                    split_id="split-derived",
                    base_events_path=historical / "base-events.json",
                    novel_events_path=historical / "novel-events.json",
                    base_records_path=historical / "base-records.json",
                    novel_records_path=historical / "novel-records.json",
                    base_train_path=historical / "base-train.json",
                    base_test_path=historical / "base-test.json",
                    validation_fraction=0.2,
                    validation_seed=20260731,
                    authority="OEPP authors",
                    output=root / "split-derived-membership.json",
                )

    def test_rejects_membership_from_a_different_source_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            membership = self._membership(source, records)
            membership["source_split"]["source_hashes"]["records.train"] = "0" * 64
            membership_path = root / "membership-wrong-source.json"
            _write_json(membership_path, membership)
            with self.assertRaisesRegex(ValueError, "source_hashes do not match"):
                import_alternate_split(
                    data_root, source, membership_path, feature_root, root / "report.json"
                )
            self.assertFalse((data_root / "splits" / "split-alternate").exists())

    def test_rejects_unhashable_provenance_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            membership = self._membership(source, records)
            membership["provenance"]["kind"] = []
            membership_path = root / "membership-invalid-kind.json"
            _write_json(membership_path, membership)
            with self.assertRaisesRegex(ValueError, "provenance.kind"):
                import_alternate_split(
                    data_root, source, membership_path, feature_root, root / "report.json"
                )
            self.assertFalse((data_root / "splits" / "split-alternate").exists())

    def test_rejects_event_leakage_without_creating_a_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            membership = self._membership(source, records)
            membership["partitions"]["validation"] = [
                {"dataset": "CrossTask", "vid": "novel-video"}
            ]
            membership["partitions"]["novel_test"] = [
                {"dataset": "COIN", "vid": "validation-video"}
            ]
            membership_path = root / "membership-event-leak.json"
            _write_json(membership_path, membership)
            with self.assertRaisesRegex(ValueError, "leaks event identities"):
                import_alternate_split(
                    data_root, source, membership_path, feature_root, root / "report.json"
                )
            self.assertFalse((data_root / "splits" / "split-alternate").exists())

    def test_rejects_missing_feature_without_creating_a_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, feature_root, records = self._source_data_root(root)
            source = SplitBundle.load(data_root, "split-source")
            membership_path = root / "membership.json"
            _write_json(membership_path, self._membership(source, records))
            (feature_root / "COIN_base-video.npy").unlink()
            with self.assertRaisesRegex(ValueError, "VideoCLIP audit failed"):
                import_alternate_split(
                    data_root, source, membership_path, feature_root, root / "report.json"
                )
            self.assertFalse((data_root / "splits" / "split-alternate").exists())


if __name__ == "__main__":
    unittest.main()
