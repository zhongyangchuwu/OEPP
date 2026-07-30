from __future__ import annotations

import unittest

import torch
from torch import nn

from oepp.models.diffusion import GaussianDiffusion


class ConstantDenoiser(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: object) -> torch.Tensor:
        return torch.full_like(x, self.value)


def build_diffusion(*, clip_denoised: bool, method: str = "uniform") -> GaussianDiffusion:
    return GaussianDiffusion(
        ConstantDenoiser(2.0),
        horizon=2,
        observation_dim=1,
        action_dim=1,
        horizon_dim=1,
        class_dim=1,
        n_timesteps=20,
        clip_denoised=clip_denoised,
        ddim_discr_method=method,
    )


class GaussianDiffusionTests(unittest.TestCase):
    def test_invalid_ddim_discretization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DDIM discretization method"):
            build_diffusion(clip_denoised=False, method="invalid")

    def test_clip_denoised_changes_sampling_reconstruction(self) -> None:
        x = torch.zeros((2, 2, 4))
        timesteps = torch.tensor([0, 1])
        clipped = build_diffusion(clip_denoised=True)
        unclipped = build_diffusion(clip_denoised=False)

        clipped_mean, _, _ = clipped.p_mean_variance(x, {}, timesteps)
        unclipped_mean, _, _ = unclipped.p_mean_variance(x, {}, timesteps)
        expected_clipped, _, _ = clipped.q_posterior(torch.ones_like(x), x, timesteps)
        expected_unclipped, _, _ = unclipped.q_posterior(torch.full_like(x, 2.0), x, timesteps)

        self.assertTrue(torch.allclose(clipped_mean, expected_clipped))
        self.assertTrue(torch.allclose(unclipped_mean, expected_unclipped))


if __name__ == "__main__":
    unittest.main()
