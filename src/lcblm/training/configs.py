"""Config dataclasses for concept model training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from lcblm.utils import get_device

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class _BaseConfig:
    epochs: int
    lr: float
    batch_size: int = 512
    seed: int = 42
    device: torch.device = field(default_factory=get_device)
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    wandb_project: str | None = None


@dataclass(frozen=True, kw_only=True)
class VAEEConfig(_BaseConfig):
    num_embeddings: int
    embedding_size: int
    pi: float
    hidden_dim: int = 256  # only used when encoder_type = "mlp"
    encoder_type: Literal["mlp", "linear", "shallow"] = "shallow"
    gumbel_temp: float = 0.5
    sigma_0: float = 0.1
    sim_metric: Literal["cosine", "inner_product", "neg_euclidean"] = "cosine"
    topology: Literal["stacked", "summed"] = "stacked"
    gamma: float = 1.0
    beta: float = 4.0
    lambda_ent: float = 1.0
    lambda_ortho: float = 0.0
    l0_threshold: float = 0.5

    def __post_init__(self) -> None:
        minimum_l0 = 2.0
        maximum_l0 = 7.0
        target_l0 = self.num_embeddings * self.pi

        if target_l0 < minimum_l0:
            warnings.warn(
                f"Your target L0 is {target_l0}, lower than {minimum_l0}.",
                stacklevel=2,
            )
        if target_l0 > maximum_l0:
            warnings.warn(
                f"Your target L0 is {target_l0}, higher than {maximum_l0}.",
                stacklevel=2,
            )


@dataclass(frozen=True, kw_only=True)
class TopKSAEConfig(_BaseConfig):
    k: int
    latent_dim: int = 0  # 0 → 4 * input_dim
    normalize_decoder: bool = False
    k_aux: int = 512
    alpha_aux: float = 1 / 32
    threshold_dead_latent: int = 1000


@dataclass(frozen=True, kw_only=True)
class SAEConceptConfig(_BaseConfig):
    vaee_num_embeddings: int
    lambda_l1: float
    normalize_decoder: bool = True


@dataclass(frozen=True, kw_only=True)
class SAEParamConfig(_BaseConfig):
    vaee_num_embeddings: int
    vaee_embedding_size: int
    lambda_l1: float
    vaee_hidden_dim: int = 256  # only used when vaee_encoder_type = "mlp"
    vaee_encoder_type: Literal["mlp", "linear", "shallow"] = "shallow"
    vaee_topology: Literal["stacked", "summed"] = "stacked"
    normalize_decoder: bool = True


@dataclass(frozen=True, kw_only=True)
class VQVAEConfig(_BaseConfig):
    """VQ-VAE dict-learning baseline (single Linear enc/dec around codebook)."""

    num_codes: int
    embedding_dim: int = 64
    commitment_weight: float = 0.25


@dataclass(frozen=True, kw_only=True)
class BetaVAEConfig(_BaseConfig):
    """β-VAE dict-learning baseline (single Linear enc/dec around z, no MLP)."""

    latent_dim: int
    beta: float
    # Threshold on |μ| for reporting how many latent dimensions are "active"
    # per sample. Disentanglement-literature convention; used for L0 reporting.
    l0_threshold: float = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    embeddings_path: str
    eos_token_id: int
    n_samples: int = -1
    name: str = ""  # defaults to config filename stem


MODEL_CONFIG_CLASSES: dict[str, type] = {
    "vaee": VAEEConfig,
    "topk_sae": TopKSAEConfig,
    "sae_concept": SAEConceptConfig,
    "sae_param": SAEParamConfig,
    "vq_vae": VQVAEConfig,
    "beta_vae": BetaVAEConfig,
}
