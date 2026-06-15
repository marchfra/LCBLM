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
    active_prob: float | None = None,
    adjacent_only: bool = False,
    coef_range: tuple[float, float] | None = None,
) -> tuple[FlatTensorDataset, Tensor]:
    """Sparse superposition benchmark.

    Each sample is a sum of active ground-truth feature vectors with random
    coefficients, plus Gaussian noise.  Returns the dataset and the ground-truth
    feature matrix (needed for feature_recovery evaluation).

    Coefficient model (the magnitude of each active atom):
    - coef_range set (e.g. (0.5, 1.5)): per-sample continuous magnitudes drawn
      from Uniform(low, high). This is the per-sample-magnitude regime — only a
      per-sample encoder can reconstruct it exactly; a fixed prototype recovers
      the atom direction but hits a magnitude-induced reconstruction floor.
      Overrides binary_coefs.
    - binary_coefs=True: coefficients are +1 on the support.
    - binary_coefs=False (default): coefficients are {-1, +1} on the support.

    Support model (which atoms are active per sample):
    - active_prob is None (default): exactly n_active atoms active per sample,
      chosen uniformly at random.
    - active_prob set: each atom is active independently with this probability
      (the canonical superposition model). The per-sample count then varies
      (Binomial(n_features, active_prob)); empty draws (k=0) are resampled so
      every sample has at least one active atom. n_active is ignored.

    adjacent_only (requires active_prob and input_dim==2): forbid co-activating
    atoms that are not neighbours on the unit circle. This restricts the support
    to singletons and ring-adjacent pairs (it caps k<=2, since on a circular
    layout any triple contains a non-adjacent pair). It removes the near-origin
    "fold-back" clusters that wide-angle pairs would otherwise produce, encoding
    a structured-sparsity prior where distant atoms never co-occur. With this
    constraint the fraction of adjacent-pair samples equals active_prob.

    min_separation: minimum angular separation (radians) between any two atoms.
    Defaults to π/n_features.  For input_dim == 2, atoms are placed at exact
    uniform angular spacing with a random rotation. For input_dim > 2, atoms
    are rejection-sampled until the constraint is met; falls back with a warning
    after _MAX_ATOM_ATTEMPTS tries.
    """
    rng = torch.Generator().manual_seed(seed)

    if adjacent_only and (active_prob is None or input_dim != 2):
        msg = "adjacent_only requires active_prob to be set and input_dim == 2"
        raise ValueError(msg)

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

    # Build the [N, n_features] support mask of active atoms per sample.
    if active_prob is None:
        # Exactly n_active atoms active per sample (uniform choice).
        rand_vals = torch.rand(n_samples, n_features, generator=rng)
        indices = rand_vals.argsort(dim=1)[:, :n_active]  # [N, n_active]
        mask = torch.zeros(n_samples, n_features)
        mask.scatter_(1, indices, 1.0)
    else:
        # Independent Bernoulli activation; resample invalid draws.  A draw is
        # invalid if it is empty (k=0) or, when adjacent_only, if its support is
        # not a singleton or a ring-adjacent pair.
        def _invalid(m: Tensor) -> Tensor:
            k = m.sum(dim=1)
            if not adjacent_only:
                return k == 0
            # Ring-adjacent active count: position i counts when i and i-1 are
            # both active (wraps around via roll).
            adj = (m * m.roll(1, dims=1)).sum(dim=1)
            valid = (k == 1) | ((k == 2) & (adj == 1))
            return ~valid

        mask = (torch.rand(n_samples, n_features, generator=rng) < active_prob).float()
        invalid = _invalid(mask)
        while bool(invalid.any()):
            n_inv = int(invalid.sum())
            mask[invalid] = (
                torch.rand(n_inv, n_features, generator=rng) < active_prob
            ).float()
            invalid = _invalid(mask)

    # Coefficients over the support mask: continuous Uniform magnitudes
    # (coef_range), {0,+1} (binary), or {-1,0,+1} (signed).
    if coef_range is not None:
        low, high = coef_range
        mags = torch.rand(n_samples, n_features, generator=rng) * (high - low) + low
        coefs = mask * mags
    elif binary_coefs:
        coefs = mask
    else:
        signs = (
            torch.randint(0, 2, (n_samples, n_features), generator=rng).float() * 2 - 1
        )
        coefs = mask * signs

    data = coefs @ features  # [N, input_dim]
    data = data + torch.randn(n_samples, input_dim, generator=rng) * noise_std

    return FlatTensorDataset(data.float()), features
