"""Model builders for concept model training."""

from __future__ import annotations

import torch  # noqa: TC002
from torch import nn  # noqa: TC002

from lcblm.baselines import BetaVAE, VQVAE
from lcblm.sae_utils import SparseAE
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.training.configs import (
    BetaVAEConfig,
    SAEParamConfig,
    TopKSAEConfig,
    VAEEConfig,
    VQVAEConfig,
)
from lcblm.utils.data import FlatTensorDataset  # noqa: TC001
from lcblm.vaee.models import VAEE


def build_vaee(input_dim: int, cfg: VAEEConfig) -> VAEE:
    return VAEE(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_embeddings=cfg.num_embeddings,
        embedding_size=cfg.embedding_size,
        gumbel_temp=cfg.gumbel_temp,
        output_activation=None,
        encoder_type=cfg.encoder_type,
        sigma_0=cfg.sigma_0,
        sim_metric=cfg.sim_metric,
        topology=cfg.topology,
    ).to(cfg.device)


def build_ref_vaee(input_dim: int, cfg: SAEParamConfig) -> VAEE:
    """Instantiate a VAEE for parameter counting only (not trained)."""
    return VAEE(
        input_dim=input_dim,
        hidden_dim=cfg.vaee_hidden_dim,
        num_embeddings=cfg.vaee_num_embeddings,
        embedding_size=cfg.vaee_embedding_size,
        gumbel_temp=0.5,
        output_activation=None,
        encoder_type=cfg.vaee_encoder_type,
        sigma_0=0.1,
        sim_metric="cosine",
        topology=cfg.vaee_topology,
    )


def build_sae(  # noqa: PLR0913
    input_dim: int,
    latent_dim: int,
    activation: nn.Module,
    train_ds: FlatTensorDataset,
    device: torch.device,
    *,
    tied_bias: bool = True,
) -> SparseAE:
    model = SparseAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        activation=activation,
        use_tied_bias=tied_bias,
    ).to(device)
    if tied_bias:
        model.init_tied_bias(compute_tied_bias(train_ds.data.cpu(), sample_every=15))
    return model


def param_matched_latent_dim(ref_vaee: VAEE, input_dim: int) -> int:
    vaee_params = sum(p.numel() for p in ref_vaee.parameters() if p.requires_grad)
    return max(1, round((vaee_params - input_dim) / (2 * input_dim)))


def resolve_latent_dim(input_dim: int, cfg: TopKSAEConfig | SAEParamConfig) -> int:
    """Resolve the effective latent_dim before training."""
    if isinstance(cfg, TopKSAEConfig):
        return cfg.latent_dim if cfg.latent_dim > 0 else 4 * input_dim
    ref = build_ref_vaee(input_dim, cfg)
    return param_matched_latent_dim(ref, input_dim)


def build_vq_vae(input_dim: int, cfg: VQVAEConfig) -> VQVAE:
    return VQVAE(
        input_dim=input_dim,
        num_codes=cfg.num_codes,
        embedding_dim=cfg.embedding_dim,
    ).to(cfg.device)


def build_beta_vae(input_dim: int, cfg: BetaVAEConfig) -> BetaVAE:
    return BetaVAE(input_dim=input_dim, latent_dim=cfg.latent_dim).to(cfg.device)
