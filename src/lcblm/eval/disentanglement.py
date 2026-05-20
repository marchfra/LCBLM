from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from torch import Tensor


def _entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def mig(
    latent_activations: Tensor,
    factors: np.ndarray,
) -> float:
    """Standard MIG (Chen et al. 2018).

    For each factor: MI gap between top-1 and top-2 latent, normalised by H(factor).
    Average over factors. Uses sklearn mutual_info_classif for MI estimates.
    """
    z = latent_activations.detach().cpu().numpy().astype(np.float32)
    n_factors = factors.shape[1]

    scores: list[float] = []
    for k in range(n_factors):
        y = factors[:, k]
        h_f = _entropy(y)
        if h_f < 1e-8:
            continue
        mi = mutual_info_classif(z, y, discrete_features=False, random_state=0)
        sorted_mi = np.sort(mi)[::-1]
        gap = sorted_mi[0] - (sorted_mi[1] if len(sorted_mi) > 1 else 0.0)
        scores.append(gap / h_f)

    return float(np.mean(scores)) if scores else 0.0


def inverted_mig(
    latent_activations: Tensor,
    factors: np.ndarray,
) -> float:
    """Inverted MIG: for each alive latent, MI gap between top-1 and top-2 factor.

    Alive = fires on ≥1% of samples (latent_activations > 0).
    Gap normalised by H(factor_top1). Average over alive latents.
    """
    z = latent_activations.detach().cpu().numpy().astype(np.float32)
    n, latent_dim = z.shape
    n_factors = factors.shape[1]

    fire_counts = (z > 0).sum(axis=0)
    alive_indices = np.where(fire_counts >= 0.01 * n)[0]

    if len(alive_indices) == 0:
        return 0.0

    scores: list[float] = []
    for j in alive_indices:
        z_j = z[:, j : j + 1]
        mi_per_factor = np.array([
            mutual_info_classif(z_j, factors[:, k], discrete_features=False, random_state=0)[0]
            for k in range(n_factors)
        ])
        top1_idx = int(np.argmax(mi_per_factor))
        sorted_mi = np.sort(mi_per_factor)[::-1]

        h_f = _entropy(factors[:, top1_idx])
        if h_f < 1e-8:
            continue

        gap = sorted_mi[0] - (sorted_mi[1] if len(sorted_mi) > 1 else 0.0)
        scores.append(gap / h_f)

    return float(np.mean(scores)) if scores else 0.0
