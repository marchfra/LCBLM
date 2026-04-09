# ruff: noqa: N803, N806

"""Model training and experiment runner."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from lcblm.embedding_ae.models import MLP, EmbeddingAE, PrototypeEmbeddingAE
from lcblm.sae_utils import SparseAE
from lcblm.sae_utils.activations import GumbelSigmoid, SigmoidCosineScoring
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.sae_utils.losses import (
    bernoulli_kl_loss_from_logits,
    bernoulli_kl_loss_from_probs,
)
from lcblm.typing import TypedLinear

from .exp_metrics import l0_embedding, l0_sparse

if TYPE_CHECKING:
    from collections.abc import Callable

    from .exp_config import DatasetConfig, RunConfig


def build_sparse_ae(n_concepts: int, cfg: RunConfig, ds_cfg: DatasetConfig) -> SparseAE:
    model = SparseAE(
        input_dim=ds_cfg.input_dim,
        latent_dim=n_concepts,
        activation=nn.ReLU(),
    )
    model.init_tied_bias(torch.empty(model.input_dim))
    return model.to(cfg.device)


def build_embedding_ae(
    n_concepts: int,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> EmbeddingAE:
    if cfg.encoder_type == "mlp":
        encoder = MLP(
            ds_cfg.input_dim,
            cfg.embedding_size * n_concepts,
            cfg.embedding_size * n_concepts,
        )
        decoder = MLP(
            cfg.embedding_size * n_concepts,
            cfg.embedding_size * n_concepts,
            ds_cfg.input_dim,
        )
    elif cfg.encoder_type == "lin":
        encoder = TypedLinear(ds_cfg.input_dim, cfg.embedding_size * n_concepts)
        decoder = TypedLinear(cfg.embedding_size * n_concepts, ds_cfg.input_dim)
    else:
        msg = "Invalid encoder type"
        raise ValueError(msg)

    _scoring_module = GumbelSigmoid(tau=cfg.tau, mu=cfg.mu)
    scoring_module = SigmoidCosineScoring(tau=cfg.tau, mu=cfg.mu)

    cls = PrototypeEmbeddingAE if cfg.decode_from_prototypes else EmbeddingAE
    return cls(
        num_embeddings=n_concepts,
        embedding_size=cfg.embedding_size,
        encoder=encoder,
        decoder=decoder,
        scoring_module=scoring_module,
    ).to(cfg.device)


def _make_sparse_ae_loss_terms(
    cfg: RunConfig,
) -> list[Callable[[SparseAE.Output], Tensor]]:
    """Construct auxiliary loss callables for SparseAE.

    Returns a list of callables each taking a SparseAE.Output and returning
    a scalar tensor. The training loop sums them with the reconstruction loss,
    with no branching at call time.
    """
    if cfg.sparsity_mode == "l1":
        return [lambda out: cfg.lambda_l1 * out.latents.abs().mean()]

    if cfg.sparsity_mode == "kl":
        return [
            lambda out: (
                cfg.lambda_kl
                * bernoulli_kl_loss_from_logits(
                    out.latents_pre_activation,
                    cfg.target_p,
                )
            ),
        ]

    msg = "Unexpected sparsity mode."
    raise ValueError(msg)


def _make_embedding_ae_loss_terms(
    model: EmbeddingAE,
    cfg: RunConfig,
) -> list[Callable[[EmbeddingAE.Output], Tensor]]:
    """Construct auxiliary loss callables for EmbeddingAE.

    Returns a list of callables each taking an EmbeddingAE.Output and
    returning a scalar tensor. Adding a new loss term means appending one
    callable here - the training loop requires no changes.
    """
    terms: list[Callable[[EmbeddingAE.Output], Tensor]] = []

    if cfg.sparsity_mode == "l1":
        terms.append(lambda out: cfg.lambda_l1 * out.scores.abs().mean())
    elif cfg.sparsity_mode == "kl":
        terms.append(
            lambda out: (
                cfg.lambda_kl * bernoulli_kl_loss_from_probs(out.scores, cfg.target_p)
            ),
        )
    else:
        msg = "Unexpected sparsity mode."
        raise ValueError(msg)

    if cfg.lambda_reg > 0.0:
        # Penalise distance between encoder embeddings and prototypes,
        # weighted by concept scores so only active concepts are pulled.
        terms.append(
            lambda out: (
                cfg.lambda_reg
                * (
                    (out.embeddings - model.prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
                    * out.scores
                ).mean()
            ),
        )

    return terms


@dataclass
class RunResult:
    """Recorded metrics for a single (model, n_concepts) training run.

    Attributes:
        model_name: "SparseAE" or "EmbeddingAE".
        n_concepts: Latent dimension / number of embeddings used.
        train_recon: Per-epoch mean reconstruction MSE on the training set.
        test_recon: Per-epoch reconstruction MSE on the test set.
        best_l0: Mean L0 evaluated on the best checkpoint.
        best_test_recon: Test reconstruction MSE of the best checkpoint.

    """

    model_name: str
    n_concepts: int
    train_recon: list[float] = field(default_factory=list)
    test_recon: list[float] = field(default_factory=list)
    best_l0: float = float("inf")
    best_test_recon: float = float("inf")


def _make_loader(X: Tensor, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)


def train_sparse_ae(
    n_concepts: int,
    X_train: Tensor,
    X_test: Tensor,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[SparseAE, RunResult]:
    """Train a SparseAE with ReLU activation and L1 regularisation.

    Args:
        n_concepts: Latent dimension.
        X_train: Normalised training tensor, shape (N_train, input_dim).
        X_test: Normalised test tensor, shape (N_test, input_dim).
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        The best-checkpoint model and the run's recorded metrics.

    """
    model = build_sparse_ae(n_concepts, cfg, ds_cfg)
    loss_terms = _make_sparse_ae_loss_terms(cfg)

    geom_median = compute_tied_bias(X_train.cpu(), 1)
    model.init_tied_bias(geom_median)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _make_loader(X_train, cfg.batch_size)
    result = RunResult(model_name="SparseAE", n_concepts=n_concepts)
    X_test_d = X_test.to(cfg.device)
    best_state: dict = {}

    for _epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_recon = 0.0
        for (xb,) in loader:
            # xb = xb.to(cfg.device)
            out = model(xb)
            recon_loss = F.mse_loss(out.recon, xb)
            loss = recon_loss + sum(term(out) for term in loss_terms)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_recon += recon_loss.item()

        result.train_recon.append(epoch_recon / len(loader))

        model.eval()
        with torch.inference_mode():
            out_test = model(X_test_d)
            test_recon = F.mse_loss(out_test.recon, X_test_d).item()
        result.test_recon.append(test_recon)

        if test_recon < result.best_test_recon:
            result.best_test_recon = test_recon
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        out_test = model(X_test_d)
        result.best_l0 = l0_sparse(out_test.latents)

    return model, result


T = TypeVar("T", int, float, torch.Tensor, np.ndarray)


class CosineAnnealing:
    def __init__(self, num_epochs: int, start_value: T, end_value: T) -> None:
        if num_epochs <= 0:
            msg = "num_epochs must be strictly positive."
            raise ValueError(msg)

        self.num_epochs = num_epochs - 1
        self.start_value = start_value
        self.end_value = end_value

    def get_value(self, epoch: int) -> T:
        return self.end_value + 0.5 * (self.start_value - self.end_value) * (
            1 + math.cos(epoch * math.pi / self.num_epochs)
        )  # ty:ignore[invalid-return-type]

    def __call__(self, epoch: int) -> T:
        return self.get_value(epoch)


def train_embedding_ae(
    n_concepts: int,
    X_train: Tensor,
    X_test: Tensor,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[EmbeddingAE, RunResult]:
    """Train an EmbeddingAE with Sigmoid scoring and L1 regularisation.

    Args:
        n_concepts: Number of prototype embeddings.
        X_train: Normalised training tensor, shape (N_train, input_dim).
        X_test: Normalised test tensor, shape (N_test, input_dim).
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        The best-checkpoint model and the run's recorded metrics.

    """
    model = build_embedding_ae(n_concepts, cfg, ds_cfg)
    loss_terms = _make_embedding_ae_loss_terms(model, cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    tau_scheduler = CosineAnnealing(cfg.epochs, cfg.tau, 0.2)
    loader = _make_loader(X_train, cfg.batch_size)
    result = RunResult(model_name="EmbeddingAE", n_concepts=n_concepts)
    X_test_d = X_test.to(cfg.device)
    best_state: dict = {}

    for epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_recon = 0.0
        for (xb,) in loader:
            # xb = xb.to(cfg.device)
            model.scoring_module.tau = tau_scheduler(epoch)
            out = model(xb)
            recon_loss = F.mse_loss(out.recon, xb)
            loss = recon_loss + sum(term(out) for term in loss_terms)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_recon += recon_loss.item()

        result.train_recon.append(epoch_recon / len(loader))

        model.eval()
        with torch.inference_mode():
            out_test = model(X_test_d)
            test_recon = F.mse_loss(out_test.recon, X_test_d).item()
        result.test_recon.append(test_recon)

        if test_recon < result.best_test_recon:
            result.best_test_recon = test_recon
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        out_test = model(X_test_d)
        result.best_l0 = l0_embedding(out_test.scores)

    return model, result


def run_experiment(
    X_train: Tensor,
    X_test: Tensor,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[list[RunResult], list[tuple[str, int, nn.Module]]]:
    """Run the full sweep over n_concepts for both model types.

    Args:
        X_train: Normalised training tensor.
        X_test: Normalised test tensor.
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        results: One RunResult per (model, n_concepts) combination.
        trained_models: Corresponding (model_name, n_concepts, model) triples,
            each holding the best checkpoint.

    """
    results: list[RunResult] = []
    trained_models: list[tuple[str, int, nn.Module]] = []

    models_to_train = (
        [(train_embedding_ae, "EmbeddingAE")]
        if cfg.skip_sae
        else [(train_sparse_ae, "SparseAE"), (train_embedding_ae, "EmbeddingAE")]
    )

    for n_concepts in cfg.n_concepts_list:
        for train_fn, label in models_to_train:
            print(f"-- {label} | n_concepts={n_concepts} --")
            model, result = train_fn(n_concepts, X_train, X_test, cfg, ds_cfg)
            trained_models.append((label, n_concepts, model))
            results.append(result)
            print(
                f"   L0={result.best_l0:.2f}  recon_MSE={result.best_test_recon:.5f}\n",
            )

    return results, trained_models
