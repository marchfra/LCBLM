"""Model training and experiment runner for the VAEE vs SparseAE experiment."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

from lcblm.sae_utils import SparseAE
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.utils.data import NextTokenDataset, typed_dataloader
from lcblm.vaee.models import VAEE, compute_loss

from .exp_metrics import l0_sparse, l0_vaee

if TYPE_CHECKING:
    from torch import Tensor

    from .exp_config import DatasetConfig, RunConfig


# ── Model builders ────────────────────────────────────────────────────────────


def build_vaee(num_embeddings: int, cfg: RunConfig, ds_cfg: DatasetConfig) -> VAEE:
    """Construct a VAEE with Identity output activation (for unbounded inputs)."""
    return VAEE(
        input_dim=ds_cfg.input_dim,
        hidden_dim=cfg.vaee_hidden_dim,
        num_embeddings=num_embeddings,
        embedding_size=cfg.vaee_embedding_size,
        gumbel_temp=cfg.vaee_gumbel_temp,
        output_activation=None,  # nn.Identity — no output range constraint
    ).to(cfg.device)


def build_sae(
    latent_dim: int,
    train_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> SparseAE:
    """Construct a SparseAE with ReLU activation and initialised tied bias."""
    model = SparseAE(
        input_dim=ds_cfg.input_dim,
        latent_dim=latent_dim,
        activation=nn.ReLU(),
    ).to(cfg.device)
    flat_train = train_ds.embeddings[train_ds.attention_mask].cpu()
    geom_median = compute_tied_bias(flat_train, sample_every=15)
    model.init_tied_bias(geom_median)
    return model


def param_matched_latent_dim(vaee: VAEE, input_dim: int) -> int:
    """Compute the SparseAE latent_dim that matches the VAEE learnable parameter count.

    SparseAE (tied_weights=True, use_tied_bias=True) has:
        2 * input_dim * latent_dim + input_dim  learnable parameters
    (encoder + decoder are separate nn.Linear layers; tied_bias is input_dim).

    Args:
        vaee: VAEE whose learnable parameters are to be matched.
        input_dim: Input feature dimension.

    Returns:
        latent_dim >= 1.

    """
    vaee_params = sum(p.numel() for p in vaee.parameters() if p.requires_grad)
    latent_dim = (vaee_params - input_dim) / (2 * input_dim)
    return max(1, round(latent_dim))


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Recorded metrics for a single (model, num_embeddings) training run.

    Attributes:
        model_name: "VAEE", "SparseAE-concept", or "SparseAE-param".
        n_concepts: num_embeddings for VAEE; latent_dim for SparseAE variants.
        train_recon: Per-epoch mean reconstruction MSE on the training set.
        val_recon: Per-epoch reconstruction MSE on the validation set.
        train_breakdown: Per-epoch dict of all training loss terms.
            VAEE keys: "recon", "cond_kl", "sparsity", "entropy".
            SparseAE keys: "recon", "l1".
        val_breakdown: Per-epoch dict of all validation loss terms (same keys).
        best_l0: Mean L0 evaluated on the best-checkpoint model.
        best_val_recon: Validation reconstruction MSE of the best checkpoint.

    """

    model_name: str
    n_concepts: int
    train_recon: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    train_breakdown: list[dict[str, float]] = field(default_factory=list)
    val_breakdown: list[dict[str, float]] = field(default_factory=list)
    best_l0: float = float("inf")
    best_val_recon: float = float("inf")
    # sweep_n is the num_embeddings value from the sweep that produced this run.
    # For VAEE and SparseAE-concept it equals n_concepts; for SparseAE-param it
    # differs (n_concepts is the parameter-matched latent_dim, sweep_n is the
    # corresponding VAEE's num_embeddings). Set to -1 until assigned by run_experiment.
    sweep_n: int = -1


# ── Helper ────────────────────────────────────────────────────────────────────


def _select_tokens(embeddings: Tensor, mask: Tensor) -> Tensor:
    """Select non-padding token embeddings using a boolean attention mask.

    Args:
        embeddings: Shape (B, L, D).
        mask: Boolean tensor of shape (B, L); True for real tokens.

    Returns:
        Shape (N_tokens, D).

    """
    return embeddings[mask]


# ── Training functions ────────────────────────────────────────────────────────


def train_vaee(  # noqa: PLR0915
    num_embeddings: int,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[VAEE, RunResult]:
    """Train a VAEE and return the best-checkpoint model with recorded metrics.

    Args:
        num_embeddings: Number of prototype embeddings (sweep axis).
        train_ds: Normalised training NextTokenDataset.
        val_ds: Normalised validation NextTokenDataset.
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        Best-checkpoint VAEE and the run's recorded metrics.

    """
    model = build_vaee(num_embeddings, cfg, ds_cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="VAEE", n_concepts=num_embeddings)
    best_state: dict = {}

    for _epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
        }
        for batch in typed_dataloader(train_loader):
            emb = batch.embeddings.to(cfg.device)
            mask = batch.attention_mask.to(cfg.device)
            tokens = _select_tokens(emb, mask)

            out = model(tokens)
            loss_out = compute_loss(
                target=tokens,
                input=out.recon,
                mu=out.mu,
                alpha=out.alpha,
                prototypes=model.prototypes,
                pi=cfg.vaee_pi,
                gamma=cfg.vaee_gamma,
                beta=cfg.vaee_beta,
                lambda_ent=cfg.vaee_lambda_ent,
            )
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            epoch_terms["recon"] += loss_out.recon_loss.item()
            epoch_terms["cond_kl"] += loss_out.cond_kl_loss.item()
            epoch_terms["sparsity"] += loss_out.sparsity_loss.item()
            epoch_terms["entropy"] += loss_out.entropy_loss.item()

        n_batches = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_batches)
        result.train_breakdown.append(
            {k: v / n_batches for k, v in epoch_terms.items()},
        )

        model.eval()
        val_terms: dict[str, float] = {
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
        }
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                emb = batch.embeddings.to(cfg.device)
                mask = batch.attention_mask.to(cfg.device)
                tokens = _select_tokens(emb, mask)
                out = model(tokens)
                loss_out = compute_loss(
                    target=tokens,
                    input=out.recon,
                    mu=out.mu,
                    alpha=out.alpha,
                    prototypes=model.prototypes,
                    pi=cfg.vaee_pi,
                    gamma=cfg.vaee_gamma,
                    beta=cfg.vaee_beta,
                    lambda_ent=cfg.vaee_lambda_ent,
                )
                val_terms["recon"] += loss_out.recon_loss.item()
                val_terms["cond_kl"] += loss_out.cond_kl_loss.item()
                val_terms["sparsity"] += loss_out.sparsity_loss.item()
                val_terms["entropy"] += loss_out.entropy_loss.item()

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        result.val_recon.append(val_recon)
        result.val_breakdown.append({k: v / n_val for k, v in val_terms.items()})

        if val_recon < result.best_val_recon:
            result.best_val_recon = val_recon
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        c_all = []
        for batch in typed_dataloader(val_loader):
            emb = batch.embeddings.to(cfg.device)
            mask = batch.attention_mask.to(cfg.device)
            c_all.append(model(_select_tokens(emb, mask)).c)
        result.best_l0 = l0_vaee(torch.cat(c_all, dim=0))

    return model, result


def train_sae(  # noqa: PLR0913
    latent_dim: int,
    model_name: str,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[SparseAE, RunResult]:
    """Train a SparseAE with ReLU + L1 sparsity.

    Args:
        latent_dim: Latent dimension (sweep axis or parameter-matched).
        model_name: "SparseAE-concept" or "SparseAE-param".
        train_ds: Normalised training NextTokenDataset.
        val_ds: Normalised validation NextTokenDataset.
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        Best-checkpoint SparseAE and the run's recorded metrics.

    """
    model = build_sae(latent_dim, train_ds, cfg, ds_cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name=model_name, n_concepts=latent_dim)
    best_state: dict = {}

    for _epoch in trange(cfg.epochs, unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {"recon": 0.0, "l1": 0.0}
        for batch in typed_dataloader(train_loader):
            emb = batch.embeddings.to(cfg.device)
            mask = batch.attention_mask.to(cfg.device)
            tokens = _select_tokens(emb, mask)

            out = model(tokens)
            recon_loss = F.mse_loss(out.recon, tokens)
            l1_loss = out.latents.abs().mean()
            loss = recon_loss + cfg.sae_lambda_l1 * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_terms["recon"] += recon_loss.item()
            epoch_terms["l1"] += l1_loss.item()

        n_batches = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_batches)
        result.train_breakdown.append(
            {k: v / n_batches for k, v in epoch_terms.items()},
        )

        model.eval()
        val_terms: dict[str, float] = {"recon": 0.0, "l1": 0.0}
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                emb = batch.embeddings.to(cfg.device)
                mask = batch.attention_mask.to(cfg.device)
                tokens = _select_tokens(emb, mask)
                out = model(tokens)
                val_terms["recon"] += F.mse_loss(out.recon, tokens).item()
                val_terms["l1"] += out.latents.abs().mean().item()

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        result.val_recon.append(val_recon)
        result.val_breakdown.append({k: v / n_val for k, v in val_terms.items()})

        if val_recon < result.best_val_recon:
            result.best_val_recon = val_recon
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        latents_all = []
        for batch in typed_dataloader(val_loader):
            emb = batch.embeddings.to(cfg.device)
            mask = batch.attention_mask.to(cfg.device)
            latents_all.append(model(_select_tokens(emb, mask)).latents)
        result.best_l0 = l0_sparse(torch.cat(latents_all, dim=0))

    return model, result


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> tuple[list[RunResult], list[tuple[str, int, nn.Module]]]:
    """Run the full sweep over num_embeddings for all model types.

    For each value in cfg.num_embeddings_list, trains:
      1. VAEE with num_embeddings = n
      2. SparseAE-concept with latent_dim = n  (concept-count-matched)
      3. SparseAE-param with latent_dim computed to match VAEE learnable parameter count

    Args:
        train_ds: Normalised training NextTokenDataset.
        val_ds: Normalised validation NextTokenDataset.
        cfg: Hyperparameter configuration.
        ds_cfg: Dataset metadata.

    Returns:
        results: One RunResult per (model, n) combination.
        trained_models: Corresponding (model_name, n_concepts, model) triples.

    """
    results: list[RunResult] = []
    trained_models: list[tuple[str, int, nn.Module]] = []

    for n in cfg.num_embeddings_list:
        print(f"\n-- VAEE | num_embeddings={n} --")
        vaee, vaee_result = train_vaee(n, train_ds, val_ds, cfg, ds_cfg)
        vaee_result.sweep_n = n
        trained_models.append(("VAEE", n, vaee))
        results.append(vaee_result)
        print(
            f"   L0={vaee_result.best_l0:.2f}  val_MSE={vaee_result.best_val_recon:.5f}",  # noqa: E501
        )

        if not cfg.skip_sae:
            print(f"-- SparseAE-concept | latent_dim={n} --")
            sae_c, sae_c_result = train_sae(
                n,
                "SparseAE-concept",
                train_ds,
                val_ds,
                cfg,
                ds_cfg,
            )
            sae_c_result.sweep_n = n
            trained_models.append(("SparseAE-concept", n, sae_c))
            results.append(sae_c_result)
            print(
                f"   L0={sae_c_result.best_l0:.2f}  val_MSE={sae_c_result.best_val_recon:.5f}",  # noqa: E501
            )

            param_dim = param_matched_latent_dim(vaee, ds_cfg.input_dim)
            print(
                f"-- SparseAE-param | latent_dim={param_dim}"
                f" (param-matched to VAEE n={n}) --",
            )
            sae_p, sae_p_result = train_sae(
                param_dim,
                "SparseAE-param",
                train_ds,
                val_ds,
                cfg,
                ds_cfg,
            )
            sae_p_result.sweep_n = (
                n  # n_concepts is param_dim, but belongs to sweep step n
            )
            trained_models.append(("SparseAE-param", n, sae_p))
            results.append(sae_p_result)
            print(
                f"   L0={sae_p_result.best_l0:.2f}  val_MSE={sae_p_result.best_val_recon:.5f}",  # noqa: E501
            )

        print()
    return results, trained_models
