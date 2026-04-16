"""Sparsity metrics for the VAEE vs SparseAE experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def l0_vaee(c: Tensor) -> float:
    """Mean number of active concepts per token for a VAEE.

    Args:
        c: Hard binary gate tensor of shape (N, num_embeddings).

    Returns:
        Mean L0 over the N tokens.

    """
    return c.float().sum(dim=1).mean().item()


def l0_sparse(latents: Tensor) -> float:
    """Mean number of active latents per token for a SparseAE.

    Counts non-zero entries in the post-activation latent tensor.

    Args:
        latents: Post-activation latent tensor of shape (N, latent_dim).

    Returns:
        Mean L0 over the N tokens.

    """
    return (latents > 0).float().sum(dim=1).mean().item()
