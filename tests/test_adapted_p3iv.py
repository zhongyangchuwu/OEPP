"""Contract tests for the OEPP-native adapted P3IV planner."""

from __future__ import annotations

import unittest

import torch

from oepp.models.p3iv_adapted import AdaptedP3IV
from oepp.models.registry import build_direct_model
from oepp.training.direct import evaluate_direct


class AdaptedP3IVTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = AdaptedP3IV(
            dimension=8,
            horizon=3,
            hidden_dimension=16,
            memory_slots=2,
            noise_dimension=4,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        self.frames = torch.randn(2, 6 * 8)

    def test_returns_one_embedding_per_planning_step(self) -> None:
        outputs = self.model.predict_with_generator(
            self.frames, generator=torch.Generator(device="cpu").manual_seed(11)
        )

        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.shape == (2, 8) for output in outputs))

    def test_fixed_generator_reproduces_prediction(self) -> None:
        self.model.eval()
        first = self.model.predict_with_generator(
            self.frames, generator=torch.Generator(device="cpu").manual_seed(19)
        )
        second = self.model.predict_with_generator(
            self.frames, generator=torch.Generator(device="cpu").manual_seed(19)
        )

        for expected, actual in zip(first, second):
            self.assertTrue(torch.equal(expected, actual))

    def test_changed_latent_changes_embedding_trajectory(self) -> None:
        self.model.eval()
        zeros = torch.zeros(2, 4)
        ones = torch.ones(2, 4)
        zero_outputs = torch.stack(self.model.predict_from_latent(self.frames, zeros), dim=1)
        one_outputs = torch.stack(self.model.predict_from_latent(self.frames, ones), dim=1)

        self.assertFalse(torch.allclose(zero_outputs, one_outputs))

    def test_diversity_penalty_is_finite_and_differentiable(self) -> None:
        loss = self.model.diversity_penalty(
            self.frames, generator=torch.Generator(device="cpu").manual_seed(23)
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.model.action_decoder.weight.grad)

    def test_seeded_validation_replays_the_same_stochastic_path(self) -> None:
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
        candidates = torch.randn(5, 8)

        first = evaluate_direct(
            self.model, loader, candidates, torch.device("cpu"), sampling_seed=29
        )
        second = evaluate_direct(
            self.model, loader, candidates, torch.device("cpu"), sampling_seed=29
        )

        self.assertEqual(first, second)

    def test_one_planner_evaluates_different_candidate_pools(self) -> None:
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
                sampling_seed=31,
            )
            self.assertEqual(set(metrics), {"sr", "acc", "mse", "miou"})

    def test_primary_decoder_is_embedding_based(self) -> None:
        self.assertEqual(self.model.primary_decoder, "semantic_embedding_cosine")

    def test_invalid_endpoint_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected endpoint frames"):
            self.model(self.frames[:, :-1])

    def test_registry_constructs_native_adaptation(self) -> None:
        model = build_direct_model(
            {
                "feature": "videoclip",
                "horizon": 3,
                "model": {
                    "family": "adapted_p3iv",
                    "hidden_dimension": 16,
                    "memory_slots": 2,
                    "noise_dimension": 4,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                }
            },
            torch.device("cpu"),
        )

        self.assertIsInstance(model, AdaptedP3IV)


if __name__ == "__main__":
    unittest.main()
