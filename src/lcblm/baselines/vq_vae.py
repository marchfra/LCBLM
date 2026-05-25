"""VQ-VAE dict-learning layer.

Structurally the hard-assignment counterpart to SparseAE: a single Linear
encoder projects the flat input into an ``embedding_dim``-dimensional code
space, the encoder output is quantized to the nearest of ``num_codes``
codebook entries (one hot, L0 = 1), and a single Linear decoder maps the
chosen code back to input space. No MLP stack, no conv stack — the Linear
projections are the dict-learning layer itself.

The straight-through estimator routes the reconstruction-loss gradient from
the decoder back through the encoder, and the commitment loss pulls encoder
outputs toward their nearest codebook entry. The codebook itself is trained
by the explicit codebook loss (Van den Oord et al. 2017).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss


class VQVAE(nn.Module):
    class Output(NamedTuple):
        recon: Tensor
        indices: Tensor
        z_q: Tensor
        z_e: Tensor

    def __init__(
        self,
        input_dim: int,
        num_codes: int,
        embedding_dim: int = 64,
    ) -> None:
        if input_dim <= 0:
            msg = "input_dim must be positive."
            raise ValueError(msg)
        if num_codes <= 0:
            msg = "num_codes must be positive."
            raise ValueError(msg)
        if embedding_dim <= 0:
            msg = "embedding_dim must be positive."
            raise ValueError(msg)

        super().__init__()
        self.input_dim = input_dim
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim

        self._encoder = nn.Linear(input_dim, embedding_dim)
        self._decoder = nn.Linear(embedding_dim, input_dim)

        # Initialise codebook with small Gaussian noise; data-aware init via
        # batch sampling is a common upgrade, but plain Gaussian is sufficient
        # for a smoke baseline and keeps the layer self-contained.
        codebook = torch.randn(num_codes, embedding_dim) * 0.1
        self.codebook = nn.Parameter(codebook)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x: Tensor) -> Tensor:
        return self._encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self._decoder(z)

    def quantize(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        """Find the nearest codebook entry for each encoder output.

        Returns:
            indices: (batch,) long tensor of chosen code indices.
            z_q: (batch, embedding_dim) selected codebook entries.

        """
        # ||z_e - c||^2 = ||z_e||^2 - 2 z_e.c + ||c||^2
        z_sq = torch.linalg.vector_norm(z_e, dim=-1, keepdim=True) ** 2  # (B, 1)
        c_sq = torch.linalg.vector_norm(self.codebook, dim=-1) ** 2  # (K,)
        cross = z_e @ self.codebook.T  # (B, K)
        dists = z_sq - 2 * cross + c_sq.unsqueeze(0)  # (B, K)

        indices = dists.argmin(dim=-1)  # (B,)
        z_q = self.codebook[indices]  # (B, embedding_dim)
        return indices, z_q

    def forward(self, x: Tensor) -> Output:
        z_e = self.encode(x)
        indices, z_q = self.quantize(z_e)

        # Straight-through estimator: forward returns z_q, but on backward
        # the gradient is routed through z_e — so the encoder learns from
        # the reconstruction loss while the codebook is updated by the
        # explicit codebook term.
        z_q_ste = z_e + (z_q - z_e).detach()
        recon = self.decode(z_q_ste)

        return self.Output(recon=recon, indices=indices, z_q=z_q, z_e=z_e)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)


class VQLossOutput(NamedTuple):
    total_loss: Tensor
    recon_loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


def compute_vq_vae_loss(
    target: Tensor,
    out: VQVAE.Output,
    commitment_weight: float = 0.25,
) -> VQLossOutput:
    """Compute the standard VQ-VAE loss.

    Args:
        target: Ground-truth input, shape (batch, input_dim).
        out: Output namedtuple from ``VQVAE.forward``.
        commitment_weight: Coefficient β on the commitment term.

    """
    recon_loss = mse_loss(out.recon, target)
    codebook_loss = mse_loss(out.z_q, out.z_e.detach())
    commitment_loss = mse_loss(out.z_e, out.z_q.detach())

    total = recon_loss + codebook_loss + commitment_weight * commitment_loss
    return VQLossOutput(
        total_loss=total,
        recon_loss=recon_loss,
        codebook_loss=codebook_loss,
        commitment_loss=commitment_loss,
    )
