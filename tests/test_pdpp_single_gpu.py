import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch

    from oepp.legacy.utils.training import (
        Trainer,
        pdpp_action_objective,
        validate_pdpp_loss_weights,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _DiffusionWithLoss(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Parameter(torch.ones(1, 1, 8))
        self.loss_calls = 0

    def loss(self, inputs: torch.Tensor, condition: dict[object, torch.Tensor]) -> torch.Tensor:
        self.loss_calls += 1
        return self.output.expand(inputs.shape[0], inputs.shape[1], -1)


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class PDPPWrapperTests(unittest.TestCase):
    def test_loss_weight_contract_rejects_invalid_objectives(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_pdpp_loss_weights(-1.0, 0.2)
        with self.assertRaisesRegex(ValueError, "at least one active"):
            validate_pdpp_loss_weights(0.0, 0.0)

    def test_loss_branches_are_independently_auditable(self) -> None:
        labels = torch.tensor([[0, 1], [1, 0]])
        train_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        for ce_weight, mse_weight, inactive_key in (
            (1.0, 0.0, "mse"),
            (0.0, 0.2, "ce"),
            (1.0, 0.2, None),
        ):
            with self.subTest(ce_weight=ce_weight, mse_weight=mse_weight):
                predicted = torch.tensor(
                    [
                        [[0.8, 0.2], [0.1, 0.9]],
                        [[0.2, 0.8], [0.9, 0.1]],
                    ],
                    requires_grad=True,
                )
                objective, metrics = pdpp_action_objective(
                    predicted,
                    labels,
                    train_embeddings,
                    ce_weight=ce_weight,
                    mse_weight=mse_weight,
                )
                self.assertAlmostEqual(
                    objective.item(), labels.shape[1] * (ce_weight + mse_weight), places=6
                )
                if inactive_key is not None:
                    self.assertIsNone(metrics[inactive_key])
                objective.backward()
                self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_single_gpu_trainer_calls_unwrapped_diffusion_loss(self) -> None:
        labels = torch.zeros((1, 3), dtype=torch.long)
        batch = [
            None,
            torch.zeros((1, 6)),
            torch.zeros((1, 6)),
            None,
            None,
            None,
            None,
            labels,
            None,
        ]
        diffusion = _DiffusionWithLoss()
        trainer = Trainer(
            diffusion,
            [batch],
            None,
            None,
            None,
            train_lr=1e-3,
            gradient_accumulate_every=1,
        )
        args = SimpleNamespace(
            class_dim=0,
            action_dim=2,
            observation_dim=6,
            horizon_dim=0,
            para_mse=0.2,
            para_ce=1.0,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        train_embeddings = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with patch.object(torch.Tensor, "cuda", new=lambda tensor, *args, **kwargs: tensor):
            trainer.train(1, False, args, scheduler, train_embeddings)

        self.assertEqual(diffusion.loss_calls, 1)
        self.assertAlmostEqual(trainer.last_loss_metrics["objective"], 1.2, places=6)
        self.assertIsNotNone(trainer.last_loss_metrics["ce"])
        self.assertIsNotNone(trainer.last_loss_metrics["mse"])


if __name__ == "__main__":
    unittest.main()
