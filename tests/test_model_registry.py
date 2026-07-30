from __future__ import annotations

import unittest

import torch

from oepp.models.registry import build_direct_model


class DirectModelRegistryTests(unittest.TestCase):
    def test_registered_mlp_returns_one_embedding_per_horizon_step(self) -> None:
        model = build_direct_model(
            {"feature": "videoclip", "horizon": 3, "model": {"family": "mlp"}},
            torch.device("cpu"),
        )
        outputs = model(torch.zeros((2, 6 * 768)))
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.shape == (2, 768) for output in outputs))

    def test_registered_transformer_returns_one_embedding_per_horizon_step(self) -> None:
        model = build_direct_model(
            {
                "feature": "videoclip",
                "horizon": 3,
                "model": {"family": "transformer", "num_layers": 1, "num_heads": 1},
            },
            torch.device("cpu"),
        )
        outputs = model(torch.zeros((2, 6 * 768)))
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.shape == (2, 768) for output in outputs))

    def test_unknown_family_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported direct OEPP model family"):
            build_direct_model(
                {"feature": "videoclip", "horizon": 3, "model": {"family": "unknown"}},
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
