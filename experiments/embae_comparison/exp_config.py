"""Experiment configuration dataclasses.

DatasetConfig holds dataset-specific metadata (dimensions, visualisation
parameters). RunConfig holds all training hyperparameters. Both are frozen so
they can be safely passed around without risk of accidental mutation.
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
        name: Short identifier used in output paths (e.g. "digits", "mnist").
        n_samples: Number of samples to load from the dataset. -1 means "all samples".
        input_dim: Flattened feature dimension fed to the models.
        img_shape: (H, W) used to reshape a flat sample for visualisation.
        img_vmax: Upper bound for imshow pixel range (16 for digits, 255 for MNIST).

    """

    name: str
    n_samples: int
    input_dim: int
    img_shape: tuple[int, int]
    img_vmax: float


@dataclass(frozen=True)
class RunConfig:
    """All training hyperparameters for a single experiment run.

    Attributes:
        n_concepts_list: Concept counts to sweep over for both models.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        tau: GumbelSigmoid's tau parameter.
        mu: GumbelSigmoid's mu parameter.
        sparsity_mode: Which sparsity penalty to apply. "l1" uses an L1 penalty
            on post-activation latents/scores; "kl" uses a Bernoulli KL penalty
            on pre-activation logits/alignments. The two modes are mutually
            exclusive.
        lambda_l1: Coefficient for the L1 sparsity term. Only used when
            sparsity_mode="l1".
        lambda_kl: Coefficient for the Bernoulli KL sparsity term. Only used
            when sparsity_mode="kl".
        target_p: Target activation probability for the Bernoulli KL loss. Must
            be in (0, 1). Only used when sparsity_mode="kl".
        embedding_size: Embedding vector size for EmbeddingAE.
        batch_size: Mini-batch size.
        top_k_examples: Number of top-activating examples per concept in the
            concept dictionary plot.
        max_concepts_in_dict: Maximum concepts shown per concept dictionary
            figure (most-active concepts are shown first).
        seed: Global random seed.
        device: Torch device to train on.
        encoder_type: The encoder/decoder to use in EmbeddingAE. Must be "lin" or "mlp".
        decode_from_prototypes: Whether to decode from the EmbeddingAE prototypes of
            from computed embeddings.

    """

    epochs: int
    lr: float
    encoder_type: Literal["lin", "mlp"] = "mlp"
    decode_from_prototypes: bool = False
    n_concepts_list: list[int] = field(
        default_factory=lambda: [5, 10, 20, 30, 50, 100],
    )
    tau: float = 1.0
    mu: float = 0.0
    sparsity_mode: Literal["l1", "kl"] = "l1"
    lambda_l1: float = 1e-2
    lambda_kl: float = 1e-2
    target_p: float = 0.05
    embedding_size: int = 8
    batch_size: int = 128
    top_k_examples: int = 5
    max_concepts_in_dict: int = 20
    seed: int = 0
    device: torch.device = field(default_factory=get_device)
