"""Contract tests for the OEPP-native adapted KEPP planner."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from oepp.baselines.kepp import build_adapted_kepp_graph
from oepp.data import SplitBundle
from oepp.models.kepp_adapted import AdaptedKEPP
from oepp.models.registry import build_direct_model
from oepp.training.direct import evaluate_direct, load_config


class AdaptedKEPPTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.node_embeddings = torch.randn(3, 8)
        self.adjacency = torch.tensor(
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.model = AdaptedKEPP(
            dimension=8,
            horizon=3,
            hidden_dimension=16,
            num_heads=2,
            graph_layers=1,
            dropout=0.0,
            node_embeddings=self.node_embeddings,
            normalized_adjacency=self.adjacency,
        )
        self.frames = torch.randn(2, 48)

    def test_train_only_graph_uses_frozen_base_pool(self) -> None:
        bundle = SplitBundle.load(Path("data"), "split-001")
        graph = build_adapted_kepp_graph(bundle, "videoclip", torch.device("cpu"))

        self.assertEqual(graph.action_names, tuple(bundle.action_pool("base")))
        self.assertEqual(graph.provenance["source_partition"], "train")
        self.assertEqual(
            graph.provenance["source_record_count"], len(bundle.partition_records("train"))
        )
        self.assertEqual(graph.node_embeddings.shape, (122, 768))
        self.assertTrue(torch.allclose(graph.normalized_adjacency.sum(dim=1), torch.ones(122)))

    def test_returns_one_embedding_per_planning_step(self) -> None:
        outputs = self.model(self.frames)

        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.shape == (2, 8) for output in outputs))

    def test_output_is_deterministic_in_evaluation_mode(self) -> None:
        self.model.eval()
        first = torch.stack(self.model(self.frames), dim=1)
        second = torch.stack(self.model(self.frames), dim=1)

        self.assertTrue(torch.equal(first, second))

    def test_graph_context_changes_output(self) -> None:
        alternate = AdaptedKEPP(
            dimension=8,
            horizon=3,
            hidden_dimension=16,
            num_heads=2,
            graph_layers=1,
            dropout=0.0,
            node_embeddings=self.node_embeddings,
            normalized_adjacency=torch.eye(3),
        )
        state = {
            name: value
            for name, value in self.model.state_dict().items()
            if name not in {"graph_node_embeddings", "normalized_adjacency"}
        }
        alternate.load_state_dict(state, strict=False)
        self.model.eval()
        alternate.eval()

        first = torch.stack(self.model(self.frames), dim=1)
        second = torch.stack(alternate(self.frames), dim=1)
        self.assertFalse(torch.allclose(first, second))

    def test_invalid_graph_normalization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows must sum to one"):
            AdaptedKEPP(
                dimension=8,
                horizon=3,
                hidden_dimension=16,
                num_heads=2,
                graph_layers=1,
                dropout=0.0,
                node_embeddings=self.node_embeddings,
                normalized_adjacency=torch.eye(3) * 2,
            )

    def test_registry_constructs_planner_with_graph_context(self) -> None:
        model = build_direct_model(
            {
                "feature": "videoclip",
                "horizon": 3,
                "model": {
                    "family": "adapted_kepp",
                    "hidden_dimension": 16,
                    "num_heads": 2,
                    "graph_layers": 1,
                    "dropout": 0.0,
                },
                "kepp_graph": {
                    "node_embeddings": torch.randn(3, 768),
                    "normalized_adjacency": self.adjacency,
                }
            },
            torch.device("cpu"),
        )

        self.assertIsInstance(model, AdaptedKEPP)

    def test_deterministic_mode_supports_different_candidate_pool_sizes(self) -> None:
        samples = [
            (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                self.frames[index],
                torch.empty(0),
                torch.randn(3, 8),
                torch.empty(0),
                torch.tensor([0, 1, 2]),
            )
            for index in range(2)
        ]
        loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)

        for candidate_count in (5, 7):
            metrics = evaluate_direct(
                self.model,
                loader,
                torch.randn(candidate_count, 8),
                torch.device("cpu"),
                prediction_mode="deterministic",
            )
            self.assertEqual(set(metrics), {"sr", "acc", "mse", "miou"})

    def test_training_configs_select_deterministic_mode(self) -> None:
        for path in (
            Path("configs/training/adapted-kepp.yaml"),
            Path("configs/training/adapted-kepp-split-002.yaml"),
        ):
            config = load_config(path)
            self.assertEqual(config["sampling"]["validation_mode"], "deterministic")
            self.assertEqual(config["sampling"]["export_mode"], "deterministic")


if __name__ == "__main__":
    unittest.main()
