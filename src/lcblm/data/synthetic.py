from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from lcblm.utils.data import FlatTensorDataset

_MAX_ATOM_ATTEMPTS = 10_000


def _place_atoms(
    n_features: int, input_dim: int, min_separation: float, rng: torch.Generator
) -> Tensor:
    """Return [n_features, input_dim] unit atoms with a min angular separation.

    input_dim == 2: exact uniform angular spacing with a random rotation.
    input_dim > 2: rejection sampling until the separation holds (warns + uses
    the last candidate after _MAX_ATOM_ATTEMPTS tries).
    """
    cos_threshold = math.cos(min_separation)
    features = torch.zeros(n_features, input_dim)

    if input_dim == 2:
        base_angle = torch.rand(1, generator=rng).item() * 2 * math.pi
        angle_step = 2 * math.pi / n_features
        for i in range(n_features):
            angle = base_angle + i * angle_step
            features[i, 0] = math.cos(angle)
            features[i, 1] = math.sin(angle)
        return features

    for i in range(n_features):
        for _attempt in range(_MAX_ATOM_ATTEMPTS):
            candidate = torch.randn(input_dim, generator=rng)
            candidate = candidate / candidate.norm().clamp(min=1e-8)
            if i == 0 or (features[:i] @ candidate).abs().max().item() < cos_threshold:
                features[i] = candidate
                break
        else:
            warnings.warn(
                f"make_synthetic: min_separation={min_separation:.4f} rad not satisfied "
                f"for atom {i} after {_MAX_ATOM_ATTEMPTS} attempts; using last candidate.",
                stacklevel=2,
            )
            features[i] = candidate  # noqa: F821  (assigned in loop body above)
    return features


def _active_mask(  # noqa: PLR0913
    n_samples: int,
    n_features: int,
    n_active: int,
    active_prob: float | None,
    rng: torch.Generator,
    *,
    adjacent_only: bool,
) -> Tensor:
    """Return the [n_samples, n_features] {0,1} support mask of active atoms.

    active_prob None: exactly n_active atoms active per sample (uniform choice).
    active_prob set: independent Bernoulli activation; invalid draws are
    resampled (empty draws always; non-singleton/non-adjacent-pair draws when
    adjacent_only).
    """
    if active_prob is None:
        rand_vals = torch.rand(n_samples, n_features, generator=rng)
        indices = rand_vals.argsort(dim=1)[:, :n_active]  # [N, n_active]
        mask = torch.zeros(n_samples, n_features)
        mask.scatter_(1, indices, 1.0)
        return mask

    def _invalid(m: Tensor) -> Tensor:
        k = m.sum(dim=1)
        if not adjacent_only:
            return k == 0
        # Ring-adjacent active count: position i counts when i and i-1 are both
        # active (wraps around via roll).
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
    return mask


def _coefficients(
    mask: Tensor,
    rng: torch.Generator,
    *,
    binary_coefs: bool,
    coef_range: tuple[float, float] | None,
) -> Tensor:
    """Per-atom anchor magnitudes over the support mask.

    coef_range: continuous Uniform(low, high); binary_coefs: {0,+1};
    else signed {-1,0,+1}.
    """
    n_samples, n_features = mask.shape
    if coef_range is not None:
        low, high = coef_range
        mags = torch.rand(n_samples, n_features, generator=rng) * (high - low) + low
        return mask * mags
    if binary_coefs:
        return mask
    signs = torch.randint(0, 2, (n_samples, n_features), generator=rng).float() * 2 - 1
    return mask * signs


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

    features = _place_atoms(n_features, input_dim, min_separation, rng)
    mask = _active_mask(
        n_samples, n_features, n_active, active_prob, rng, adjacent_only=adjacent_only
    )
    coefs = _coefficients(mask, rng, binary_coefs=binary_coefs, coef_range=coef_range)

    data = coefs @ features  # [N, input_dim]
    data = data + torch.randn(n_samples, input_dim, generator=rng) * noise_std

    return FlatTensorDataset(data.float()), features


def _make_subspaces(
    features: Tensor, subspace_rank: int, rng: torch.Generator
) -> Tensor:
    """Return [n_features, subspace_rank, input_dim] per-concept subspace bases.

    For each concept i, B_i is an orthonormal basis of a random subspace of the
    space *orthogonal to its own anchor* features[i]. Across concepts the bases
    are independent random draws (not mutually orthogonal — impossible once
    n_features + n_features*rank exceeds input_dim). In 2D with rank 1 the only
    direction orthogonal to the anchor is unique (up to sign), so B_i is the
    tangent to the unit circle at the anchor.
    """
    n_features, input_dim = features.shape
    if subspace_rank >= input_dim:
        msg = f"subspace_rank ({subspace_rank}) must be < input_dim ({input_dim})"
        raise ValueError(msg)

    bases = torch.zeros(n_features, subspace_rank, input_dim)
    for i in range(n_features):
        a = features[i]
        # Random vectors, projected onto the orthogonal complement of the anchor,
        # then orthonormalised via QR.
        v = torch.randn(input_dim, subspace_rank, generator=rng)
        v = v - torch.outer(a, a @ v)  # remove the anchor component (column-wise)
        q, _ = torch.linalg.qr(v)  # [input_dim, subspace_rank], orthonormal cols
        bases[i] = q.T
    return bases


def make_complex_synthetic(  # noqa: PLR0913
    n_samples: int = 20_000,
    n_features: int = 64,
    n_active: int = 5,
    input_dim: int = 32,
    noise_std: float = 0.05,
    seed: int = 0,
    *,
    binary_coefs: bool = True,
    min_separation: float | None = None,
    active_prob: float | None = None,
    adjacent_only: bool = False,
    coef_range: tuple[float, float] | None = None,
    subspace_rank: int = 1,
    subspace_scale: float = 0.3,
) -> tuple[FlatTensorDataset, Tensor]:
    """Structured intra-concept-variance benchmark (mixture of factor analysers).

    Each concept i is no longer a single ray but an anchor a_i plus an
    r-dimensional affine subspace B_i. When concept i is active in a sample, its
    contribution is

        coef_i * a_i  +  subspace_scale * (w_i @ B_i),   w_i ~ N(0, I_r)

    so every sample of the concept lands at a different point of the concept's
    (r-dimensional) flat around a_i. B_i is orthogonal to a_i (see
    _make_subspaces), so the variation is in genuinely new directions a 1-D
    dictionary atom cannot reach.

    This is the regime that should favour a subspace concept (VAEE, one E-dim
    concept with E >= r+1) over a 1-D SAE latent, which must spend ~r+1
    co-firing latents to cover one concept's subspace.

    Returns (dataset, anchors). Anchors are the ground-truth a_i used by
    feature_recovery; reconstruction MSE is the headline metric for whether a
    model captures the within-concept subspace. All other arguments match
    make_synthetic.
    """
    rng = torch.Generator().manual_seed(seed)

    if adjacent_only and (active_prob is None or input_dim != 2):
        msg = "adjacent_only requires active_prob to be set and input_dim == 2"
        raise ValueError(msg)

    if min_separation is None:
        min_separation = math.pi / n_features

    features = _place_atoms(n_features, input_dim, min_separation, rng)
    bases = _make_subspaces(features, subspace_rank, rng)  # [K, r, D]
    mask = _active_mask(
        n_samples, n_features, n_active, active_prob, rng, adjacent_only=adjacent_only
    )
    coefs = _coefficients(mask, rng, binary_coefs=binary_coefs, coef_range=coef_range)

    # Anchor part: sum_i coef_i a_i.
    anchor_part = coefs @ features  # [N, D]

    # Subspace part: sum over active concepts of scale * (w_i @ B_i). Mask the
    # per-sample latents so only active concepts contribute, and contract k, r in
    # one einsum (no [N, K, D] intermediate).
    w = torch.randn(n_samples, n_features, subspace_rank, generator=rng)
    wm = mask.unsqueeze(-1) * w  # [N, K, r]
    subspace_part = subspace_scale * torch.einsum("nkr,krd->nd", wm, bases)

    data = anchor_part + subspace_part
    data = data + torch.randn(n_samples, input_dim, generator=rng) * noise_std

    return FlatTensorDataset(data.float()), features
