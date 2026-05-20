"""Unit tests for MIG and inverted-MIG implementations."""

from __future__ import annotations

import numpy as np
import torch

from lcblm.eval.disentanglement import inverted_mig, mig


def _perfect_latents(n: int = 2000, n_factors: int = 4) -> tuple[torch.Tensor, np.ndarray]:
    """Each latent i is a noisy copy of factor i; factors are discrete uniform."""
    rng = np.random.default_rng(0)
    factors = rng.integers(0, 8, size=(n, n_factors)).astype(np.int64)
    # Latent j = factor_j + small noise → near-perfect correlation.
    z = factors.astype(np.float32) + rng.normal(0, 0.05, size=(n, n_factors)).astype(np.float32)
    # Ensure all latents are positive so alive check passes.
    z = z - z.min() + 0.1
    return torch.from_numpy(z), factors


def _random_latents(n: int = 2000, latent_dim: int = 8, n_factors: int = 4) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(1)
    factors = rng.integers(0, 8, size=(n, n_factors)).astype(np.int64)
    z = rng.uniform(0, 1, size=(n, latent_dim)).astype(np.float32)
    return torch.from_numpy(z), factors


def test_mig_perfect_correlation():
    z, factors = _perfect_latents()
    score = mig(z, factors)
    assert score >= 0.9, f"Expected MIG ≥ 0.9 for perfectly correlated latents, got {score:.4f}"


def test_mig_random_noise():
    z, factors = _random_latents()
    score = mig(z, factors)
    assert score <= 0.05, f"Expected MIG ≤ 0.05 for random latents, got {score:.4f}"


def test_inverted_mig_perfect_correlation():
    z, factors = _perfect_latents()
    score = inverted_mig(z, factors)
    assert score >= 0.9, f"Expected inverted MIG ≥ 0.9 for perfectly correlated latents, got {score:.4f}"


def test_inverted_mig_random_noise():
    z, factors = _random_latents()
    score = inverted_mig(z, factors)
    assert score <= 0.05, f"Expected inverted MIG ≤ 0.05 for random latents, got {score:.4f}"
