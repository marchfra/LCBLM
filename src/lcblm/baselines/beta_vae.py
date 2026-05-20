"""β-VAE dict-learning layer.

Minimal Gaussian VAE with a β-weighted KL: a single linear encoder produces
(μ, log σ²) over a latent of size ``latent_dim``; a single linear decoder maps z
back to the input. No MLP stack, no conv stack — the Linear projections are
the dict-learning layer itself, mirroring the structure of
``lcblm.sae_utils.SparseAE``.

The β coefficient controls the disentanglement-vs-reconstruction trade-off
(Higgins et al. 2017). It is the Pareto-sweep axis for this baseline.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss


class BetaVAE(nn.Module):
    class Output(NamedTuple):
        recon: Tensor
        mu: Tensor
        logvar: Tensor
        z: Tensor

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
    ) -> None:
        if input_dim <= 0:
            msg = "input_dim must be positive."
            raise ValueError(msg)
        if latent_dim <= 0:
            msg = "latent_dim must be positive."
            raise ValueError(msg)

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder maps x → (μ, logσ²) as a single linear projection.
        self._encoder = nn.Linear(input_dim, 2 * latent_dim)
        self._decoder = nn.Linear(latent_dim, input_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        params = self._encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor) -> Tensor:
        return self._decoder(z)

    def forward(self, x: Tensor) -> Output:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return self.Output(recon=recon, mu=mu, logvar=logvar, z=z)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)


class BetaVAELossOutput(NamedTuple):
    total_loss: Tensor
    recon_loss: Tensor
    kl_loss: Tensor


def compute_beta_vae_loss(
    target: Tensor,
    out: BetaVAE.Output,
    beta: float,
) -> BetaVAELossOutput:
    """Compute the β-VAE ELBO with a configurable KL weight.

    KL is reduced as the per-sample sum over latent dimensions, then averaged
    over the batch — matching the standard formulation (Higgins et al. 2017).
    The reconstruction term is MSE for continuous-valued inputs.

    Args:
        target: Ground-truth input, shape (batch, input_dim).
        out: Output namedtuple from ``BetaVAE.forward``.
        beta: Coefficient on the KL term.

    """
    recon_loss = mse_loss(out.recon, target)

    # KL(N(μ, σ²) || N(0, I)) = -0.5 · Σ (1 + logσ² - μ² - σ²)
    kl_per_sample = -0.5 * torch.sum(
        1 + out.logvar - out.mu.pow(2) - out.logvar.exp(),
        dim=-1,
    )
    kl_loss = kl_per_sample.mean()

    total = recon_loss + beta * kl_loss
    return BetaVAELossOutput(
        total_loss=total,
        recon_loss=recon_loss,
        kl_loss=kl_loss,
    )
