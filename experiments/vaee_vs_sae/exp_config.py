"""Experiment configuration dataclasses.

DatasetConfig holds dataset-specific metadata. RunConfig holds all training
hyperparameters. Both are frozen so they can be safely passed around without
risk of accidental mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from lcblm.utils import get_device

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific metadata used across training and plotting.

    Attributes:
        name: Short identifier used in output paths (e.g. "sst2_mistral").
        embeddings_path: Path to the directory containing the extracted_features_*.pt
            files. Each file is a dict with keys input_ids, attention_masks, embeddings,
            all of shape (num_sentences, context_window, input_dim).
        input_dim: Embedding dimension fed to the models (e.g. 4096 for Mistral-7B).
        eos_token_id: EOS token ID used to construct NextTokenDataset.
        n_samples: Number of sentences to load per split. -1 means all sentences.

    """

    name: str
    embeddings_path: str
    input_dim: int
    eos_token_id: int
    n_samples: int = -1


@dataclass(frozen=True)
class RunConfig:
    """All training hyperparameters for a single experiment run.

    Attributes:
        num_embeddings_list: Number of VAEE embeddings (and SAE concepts) to sweep over.
        epochs: Number of training epochs per model.
        lr: Adam learning rate.
        batch_size: Mini-batch size (in sentences; tokens are extracted per batch).
        seed: Global random seed.
        device: Torch device to train on.
        vaee_hidden_dim: Hidden layer size for the VAEE encoder/decoder MLPs. Should be
            much smaller than input_dim to act as a bottleneck. Ignored when vaee_linear
            is True.
        vaee_encoder_type: Encoder/decoder architecture. "mlp" uses a 2-layer MLP with
            hidden_dim bottleneck (default). "linear" uses a single nn.Linear (no
            bottleneck, no non-linearity). "shallow" uses nn.Linear + GELU (no
            bottleneck, with non-linearity). hidden_dim is ignored for "linear" and
            "shallow".
        vaee_embedding_size: Dimensionality of each prototype embedding vector.
        vaee_gumbel_temp: Gumbel-Sigmoid temperature (lower = more discrete).
        vaee_pi: Target Bernoulli activation probability for VAEE sparsity loss.
        vaee_gamma: Coefficient for the conditional KL loss term.
        vaee_beta: Coefficient for the sparsity KL loss term.
        vaee_beta_warmup_epochs: Number of epochs over which beta is linearly ramped
            from 0 to vaee_beta. 0 disables warmup (beta is constant throughout).
        vaee_lambda_ent: Coefficient for the entropy regularisation term.
        sae_lambda_l1: L1 sparsity coefficient for both SparseAE variants.
        sae_normalize_decoder: If True, apply decoder column normalisation during
            SparseAE training: gradient projection before optimizer step (keeps Adam
            momentum clean) and re-normalisation after (corrects floating-point drift).
            Fixes the shrinkage problem where encoder weights shrink and decoder weights
            grow to reduce L1 without actually increasing sparsity.
        skip_vaee: If True, skip training the VAEE.
        skip_sae: If True, skip training both SparseAE variants.
        wandb_project: W&B project name. If None, W&B logging is disabled.
            Use a separate project for exploratory runs (e.g. "vaee-vs-sae-dev")
            to keep the main project clean.

    """

    epochs: int
    lr: float
    num_embeddings_list: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256],
    )
    batch_size: int = 512
    seed: int = 42
    device: torch.device = field(default_factory=get_device)
    vaee_hidden_dim: int = 256
    vaee_encoder_type: Literal["mlp", "linear", "shallow"] = "mlp"
    vaee_embedding_size: int = 16
    vaee_gumbel_temp: float = 0.5
    vaee_pi: float = 0.1
    vaee_gamma: float = 0.01
    vaee_beta: float = 1.0
    vaee_beta_warmup_epochs: int = 0
    vaee_lambda_ent: float = 0.01
    sae_lambda_l1: float = 1e-3
    sae_normalize_decoder: bool = False
    skip_vaee: bool = False
    skip_sae: bool = False
    wandb_project: str | None = None
