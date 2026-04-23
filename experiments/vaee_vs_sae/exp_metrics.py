"""Sparsity metrics for the VAEE vs SparseAE experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def l0_vaee(c: Tensor, threshold: float = 1e-6) -> float:
    """Mean number of active concepts per token for a VAEE.

    A concept is considered active when its gate value exceeds 0.5.
    At eval time c = alpha = sigmoid(logits), which is continuous in (0, 1),
    so thresholding is necessary to obtain a meaningful integer count.

    Args:
        c: Gate tensor of shape (N, num_embeddings), values in (0, 1).
        threshold: Minimum threshold to consider an embedding active.

    Returns:
        Mean L0 over the N tokens.

    """
    return (c > threshold).float().sum(dim=1).mean().item()


def l0_sparse(latents: Tensor) -> float:
    """Mean number of active latents per token for a SparseAE.

    Counts non-zero entries in the post-activation latent tensor.

    Args:
        latents: Post-activation latent tensor of shape (N, latent_dim).

    Returns:
        Mean L0 over the N tokens.

    """
    return (latents > 0).float().sum(dim=1).mean().item()
