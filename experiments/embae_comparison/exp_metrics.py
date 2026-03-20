"""Sparsity metrics for SparseAE and EmbeddingAE."""

from __future__ import annotations

import torch
from torch import Tensor


def l0_sparse(latents: Tensor) -> float:
    """Mean number of non-zero latents per sample (SparseAE).

    Args:
        latents: Post-activation latent tensor, shape (N, latent_dim).

    Returns:
        Mean L0 across the batch.

    """
    return (latents > 0).float().sum(-1).mean().item()


def l0_embedding(scores: Tensor) -> float:
    """Mean number of active concepts per sample after thresholding (EmbeddingAE).

    Scores are binarised at 0.5 before counting, since sigmoid outputs are
    continuous in (0, 1).

    Args:
        scores: Per-concept sigmoid scores, shape (N, num_embeddings).

    Returns:
        Mean L0 across the batch.

    """
    threshold = 0.5
    binarized = torch.where(
        scores > threshold,
        torch.ones_like(scores),
        torch.zeros_like(scores),
    )
    return binarized.sum(-1).mean().item()
