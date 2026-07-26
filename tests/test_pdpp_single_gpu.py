import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch

    from utils.training import Trainer

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


if __name__ == "__main__":
    unittest.main()
