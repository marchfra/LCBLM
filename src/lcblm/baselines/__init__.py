"""Dict-learning baselines for the workshop paper.

Each baseline operates directly on flat pixel/vector input — no separate encoder
or decoder is wrapped around the dict-learning layer. See
`experiments/dict_learning_paper/PLAN.md` for the full design rationale.
"""

from lcblm.baselines.beta_vae import BetaVAE, compute_beta_vae_loss
from lcblm.baselines.vq_vae import VQVAE, compute_vq_vae_loss

__all__ = [
    "BetaVAE",
    "VQVAE",
    "compute_beta_vae_loss",
    "compute_vq_vae_loss",
]
