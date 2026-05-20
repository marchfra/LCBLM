from __future__ import annotations

import torch
from torch import Tensor

from lcblm.utils.data import FlatTensorDataset


def make_synthetic(
    n_samples: int = 50_000,
    n_features: int = 512,
    n_active: int = 5,
    input_dim: int = 256,
    noise_std: float = 0.1,
    seed: int = 0,
) -> tuple[FlatTensorDataset, Tensor]:
    """Sparse superposition benchmark.

    Each sample is a sum of n_active ground-truth feature vectors with random ±1
    coefficients, plus Gaussian noise.  Returns the dataset and the ground-truth
    feature matrix (needed for feature_recovery evaluation).
    """
    rng = torch.Generator().manual_seed(seed)

    features = torch.randn(n_features, input_dim, generator=rng)
    features = features / features.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Vectorised: for each sample, random-shuffle feature indices and take first n_active.
    rand_vals = torch.rand(n_samples, n_features, generator=rng)
    indices = rand_vals.argsort(dim=1)[:, :n_active]  # [N, n_active]

    signs = torch.randint(0, 2, (n_samples, n_active), generator=rng).float() * 2 - 1
    selected = features[indices]  # [N, n_active, input_dim]
    data = (signs.unsqueeze(-1) * selected).sum(dim=1)  # [N, input_dim]
    data = data + torch.randn(n_samples, input_dim, generator=rng) * noise_std

    return FlatTensorDataset(data.float()), features
