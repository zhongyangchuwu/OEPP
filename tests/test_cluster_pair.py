from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from oepp.data import ActionPool, Partition, SplitBundle
from oepp.data.cluster_pair import (
    PAIR_PLAN_SCHEMA,
    _ordered_pools,
    build_fold_assignments,
    cluster_folds,
)
from oepp.evaluation.cluster_pair import summarize_cluster_pair
from oepp.experiments.cluster_pair import build_tasks


class ClusterPairSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = SplitBundle.load(Path("data"), "split-001")
        cls.folds = cluster_folds(cls.bundle, anchor_seed=20260810)

    def test_task_info_clusters_produce_fourteen_paired_folds(self) -> None:
        self.assertEqual(len(self.folds), 14)
        self.assertEqual(sum(len(fold.target_events) for fold in self.folds), 29)
        first = next(fold for fold in self.folds if fold.cluster_id == "0")
        self.assertEqual(first.anchor_event, "BandageDogPaw")
        self.assertEqual(first.target_events, ("BandageHead",))

    def test_blocked_condition_excludes_only_anchor_and_keeps_targets_identical(self) -> None:
        fold = next(fold for fold in self.folds if fold.cluster_id == "0")
        assignments = build_fold_assignments(self.bundle, fold, partition_seed=20260810)
        supported = assignments["supported"]
        blocked = assignments["blocked"]
        self.assertEqual(len(supported["excluded"]), 0)
        self.assertTrue(blocked["excluded"])
        self.assertEqual(
            {record["task_name"] for record in blocked["excluded"]},
            {fold.anchor_event},
        )
        supported_novel = supported["partitions"][Partition.NOVEL_TEST]
        blocked_novel = blocked["partitions"][Partition.NOVEL_TEST]
        self.assertEqual(
            {(record["dataset"], record["vid"]) for record in supported_novel},
            {(record["dataset"], record["vid"]) for record in blocked_novel},
        )
        self.assertEqual(
            _ordered_pools(self.bundle, supported["partitions"])[ActionPool.NOVEL],
            _ordered_pools(self.bundle, blocked["partitions"])[ActionPool.NOVEL],
        )
        for condition in (supported, blocked):
            for partition in Partition:
                self.assertTrue(condition["partitions"][partition])


class ClusterPairSummaryTests(unittest.TestCase):
    fields = [
        "sample_id",
        "dataset",
        "task_name",
        "task_id",
        "vid",
        "start_step",
        "end_step",
        "step_index",
        "gt_action",
        "predicted_action",
        "correct",
    ]

    @classmethod
    def _write_rows(cls, path: Path, *, event: str, blocked: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for window in range(2):
            for step in range(3):
                correct = not blocked or step > 0
                rows.append(
                    {
                        "sample_id": f"{event}-{window}",
                        "dataset": "crosstask",
                        "task_name": event,
                        "task_id": event,
                        "vid": f"video-{window}",
                        "start_step": "0",
                        "end_step": "2",
                        "step_index": str(step),
                        "gt_action": f"action-{step}",
                        "predicted_action": f"action-{step}" if correct else "wrong",
                        "correct": "1" if correct else "0",
                    }
                )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_summary_pairs_events_and_applies_directional_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {
                "schema": PAIR_PLAN_SCHEMA,
                "protocol_id": "test-cluster-pair",
                "claim_boundary": "test boundary",
                "folds": [],
            }
            for index, event in enumerate(("event-a", "event-b")):
                plan["folds"].append(
                    {
                        "fold_index": index,
                        "cluster_id": str(index),
                        "anchor_event": f"anchor-{index}",
                        "target_events": [event],
                        "splits": {
                            "supported": {"split_id": f"s-{index}"},
                            "blocked": {"split_id": f"b-{index}"},
                        },
                    }
                )
                for condition in ("supported", "blocked"):
                    self._write_rows(
                        root
                        / "runs"
                        / "mlp"
                        / f"fold-{index:02d}"
                        / condition
                        / "exports"
                        / "novel_metrics_per_window_step.csv",
                        event=event,
                        blocked=condition == "blocked",
                    )
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report = summarize_cluster_pair(
                plan_path=plan_path,
                run_root=root,
                models=("mlp",),
                bootstrap_seed=7,
                bootstrap_repeats=100,
            )
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["models"]["mlp"]["target_events"], 2)
            self.assertLess(
                report["models"]["mlp"]["blocked_minus_supported"]["event_macro"]["sr"],
                0,
            )

    def test_runner_builds_two_conditions_per_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = {
                "schema": PAIR_PLAN_SCHEMA,
                "protocol_id": "test-cluster-pair",
                "folds": [
                    {
                        "fold_index": 0,
                        "splits": {
                            "supported": {"split_id": "supported"},
                            "blocked": {"split_id": "blocked"},
                        },
                    }
                ],
            }
            tasks = build_tasks(
                plan=plan,
                output_root=Path(temporary),
                models=("mlp", "transformer"),
                epochs=3,
            )
            self.assertEqual(len(tasks), 4)
            self.assertEqual({task.condition for task in tasks}, {"supported", "blocked"})
            for task in tasks:
                config = task.config_path.read_text(encoding="utf-8")
                self.assertIn("epochs: 3", config)
                self.assertIn(f"split_id: {task.split_id}", config)


if __name__ == "__main__":
    unittest.main()
