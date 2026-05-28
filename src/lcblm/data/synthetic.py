from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from lcblm.utils.data import FlatTensorDataset

_MAX_ATOM_ATTEMPTS = 10_000


def make_synthetic(  # noqa: PLR0913
    n_samples: int = 50_000,
    n_features: int = 512,
    n_active: int = 5,
    input_dim: int = 256,
    noise_std: float = 0.1,
    seed: int = 0,
    *,
    binary_coefs: bool = False,
    min_separation: float | None = None,
) -> tuple[FlatTensorDataset, Tensor]:
    """Sparse superposition benchmark.

    Each sample is a sum of n_active ground-truth feature vectors with random
    coefficients drawn from {-1, +1} (default) or {+1} when binary_coefs=True,
    plus Gaussian noise.  Returns the dataset and the ground-truth feature
    matrix (needed for feature_recovery evaluation).

    min_separation: minimum angular separation (radians) between any two atoms.
    Defaults to π/n_features.  For input_dim == 2, atoms are placed at exact
    uniform angular spacing with a random rotation. For input_dim > 2, atoms
    are rejection-sampled until the constraint is met; falls back with a warning
    after _MAX_ATOM_ATTEMPTS tries.
    """
    rng = torch.Generator().manual_seed(seed)

    if min_separation is None:
        min_separation = math.pi / n_features
    cos_threshold = math.cos(min_separation)

    features = torch.zeros(n_features, input_dim)

    if input_dim == 2:
        # Analytic uniform spacing for 2D
        base_angle = torch.rand(1, generator=rng).item() * 2 * math.pi
        angle_step = 2 * math.pi / n_features
        for i in range(n_features):
            angle = base_angle + i * angle_step
            features[i, 0] = math.cos(angle)
            features[i, 1] = math.sin(angle)
    else:
        # Rejection sampling for higher dimensions
        for i in range(n_features):
            for attempt in range(_MAX_ATOM_ATTEMPTS):
                candidate = torch.randn(input_dim, generator=rng)
                candidate = candidate / candidate.norm().clamp(min=1e-8)
                if (
                    i == 0
                    or (features[:i] @ candidate).abs().max().item() < cos_threshold
                ):
                    features[i] = candidate
                    break
            else:
                warnings.warn(
                    f"make_synthetic: min_separation={min_separation:.4f} rad not satisfied "
                    f"for atom {i} after {_MAX_ATOM_ATTEMPTS} attempts; using last candidate.",
                    stacklevel=2,
                )
                features[i] = candidate  # noqa: F821  (assigned in loop body above)

    # Vectorised: for each sample, random-shuffle feature indices and take first
    # n_active.
    rand_vals = torch.rand(n_samples, n_features, generator=rng)
    indices = rand_vals.argsort(dim=1)[:, :n_active]  # [N, n_active]

    if binary_coefs:
        coefs = torch.ones(n_samples, n_active)
    else:
        coefs = (
            torch.randint(0, 2, (n_samples, n_active), generator=rng).float() * 2 - 1
        )
    selected = features[indices]  # [N, n_active, input_dim]
    data = (coefs.unsqueeze(-1) * selected).sum(dim=1)  # [N, input_dim]
    data = data + torch.randn(n_samples, input_dim, generator=rng) * noise_std

    return FlatTensorDataset(data.float()), features
