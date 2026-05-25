from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def alive_dict_size(
    latent_activations: Tensor,
    threshold: float = 0.001,
) -> int:
    """Count latent dimensions that fire on at least threshold x N samples."""
    n = latent_activations.shape[0]
    fire_counts = (latent_activations > 0).float().sum(0)  # [latent_dim]
    return int((fire_counts >= threshold * n).sum().item())


def class_purity(
    latent_activations: Tensor,
    labels: Tensor,
    top_k: int = 50,
) -> Tensor:
    """Per-concept label purity from the top-k activating samples.

    Returns a [latent_dim] tensor where each entry is 1 - H/H_max.
    """
    n, latent_dim = latent_activations.shape
    n_classes = int(labels.max().item()) + 1
    h_max = math.log(n_classes) if n_classes > 1 else 1.0
    k = min(top_k, n)

    purity = torch.zeros(latent_dim)
    for j in range(latent_dim):
        top_idx = latent_activations[:, j].topk(k).indices
        top_labels = labels[top_idx]
        counts = torch.bincount(top_labels, minlength=n_classes).float()
        probs = counts / counts.sum()
        h = -(probs * probs.clamp(min=1e-12).log()).sum().item()
        purity[j] = 1.0 - h / h_max if h_max > 0 else 1.0

    return purity


def feature_recovery(
    learned_directions: Tensor,
    ground_truth_features: Tensor,
    threshold: float = 0.9,
) -> dict[str, float]:
    """Hungarian-matched cosine similarity between learned and ground-truth directions.

    Returns matched_fraction (fraction of GT features matched above threshold)
    and mean_cosine_sim (mean over all matched pairs).
    """
    ld = F.normalize(learned_directions, dim=1)  # [latent_dim, input_dim]
    gt = F.normalize(ground_truth_features, dim=1)  # [n_features, input_dim]
    cos_sim = ld @ gt.T  # [latent_dim, n_features]

    row_idx, col_idx = linear_sum_assignment(-cos_sim.detach().cpu().numpy())
    matched_sims = cos_sim[row_idx, col_idx]

    n_features = ground_truth_features.shape[0]
    above = int((matched_sims >= threshold).sum().item())

    return {
        "matched_fraction": above / n_features,
        "mean_cosine_sim": float(matched_sims.mean().item()),
    }
