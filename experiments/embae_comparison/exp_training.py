# ruff: noqa: N803, N806

"""Model training and experiment runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from lcblm.embedding_ae import EmbeddingAE
from lcblm.embedding_ae.models import MLP
from lcblm.sae_utils import SparseAE
from lcblm.sae_utils.activations import GumbelSigmoid
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.sae_utils.losses import bernoulli_kl_loss
from lcblm.typing import TypedLinear

from .exp_metrics import l0_embedding, l0_sparse

if TYPE_CHECKING:
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

    return EmbeddingAE(
        num_embeddings=n_concepts,
        embedding_size=cfg.embedding_size,
        encoder=encoder,
        decoder=decoder,
        scoring_module=GumbelSigmoid(tau=cfg.tau, mu=cfg.mu),
        decode_from_prototypes=cfg.decode_from_prototypes,
    ).to(cfg.device)


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
    print(model)
    print(model.device)

    geom_median = compute_tied_bias(X_train, 1)
    model.init_tied_bias(geom_median)
    print(model.tied_bias)
    print(model.tied_bias.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _make_loader(X_train, cfg.batch_size)
    result = RunResult(model_name="SparseAE", n_concepts=n_concepts)
    X_test_d = X_test.to(cfg.device)
    best_state: dict = {}

    for _epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_recon = 0.0
        for (xb,) in loader:
            xb = xb.to(cfg.device)  # noqa: PLW2901
            out = model(xb)
            recon_loss = F.mse_loss(out.recon, xb)
            if cfg.sparsity_mode == "l1":
                sparsity_loss = cfg.lambda_l1 * out.latents.abs().mean()
            elif cfg.sparsity_mode == "kl":
                sparsity_loss = cfg.lambda_kl * bernoulli_kl_loss(
                    out.latents_pre_activation,
                    cfg.target_p,
                )
            else:
                msg = "Invalid sparsity mode"
                raise ValueError(msg)
            loss = recon_loss + sparsity_loss
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

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loader = _make_loader(X_train, cfg.batch_size)
    result = RunResult(model_name="EmbeddingAE", n_concepts=n_concepts)
    X_test_d = X_test.to(cfg.device)
    best_state: dict = {}

    for _epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_recon = 0.0
        for (xb,) in loader:
            xb = xb.to(cfg.device)  # noqa: PLW2901
            out = model(xb)
            recon_loss = F.mse_loss(out.recon, xb)
            if cfg.sparsity_mode == "l1":
                sparsity_loss = cfg.lambda_l1 * out.scores.abs().mean()
            elif cfg.sparsity_mode == "kl":
                sparsity_loss = cfg.lambda_kl * bernoulli_kl_loss(
                    out.alignments,
                    cfg.target_p,
                )
            else:
                msg = "Invalid sparsity mode"
                raise ValueError(msg)
            loss = recon_loss + sparsity_loss
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

    for n_concepts in cfg.n_concepts_list:
        for train_fn, label in (
            (train_sparse_ae, "SparseAE"),
            (train_embedding_ae, "EmbeddingAE"),
        ):
            print(f"-- {label} | n_concepts={n_concepts} --")
            model, result = train_fn(n_concepts, X_train, X_test, cfg, ds_cfg)
            trained_models.append((label, n_concepts, model))
            results.append(result)
            print(
                f"   L0={result.best_l0:.2f}  recon_MSE={result.best_test_recon:.5f}\n",
            )

    return results, trained_models
