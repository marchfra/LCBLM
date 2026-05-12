"""Training loops for VAEE, TopK-SAE, and L1-SAE concept models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

import wandb  # noqa: TC001
from lcblm.sae_utils import SparseAE, TopK
from lcblm.sae_utils.activations import update_dead_latent_counts
from lcblm.sae_utils.losses import loss_k_aux, loss_top_k
from lcblm.training.configs import (  # noqa: TC001
    SAEConceptConfig,
    SAEParamConfig,
    TopKSAEConfig,
    VAEEConfig,
)
from lcblm.training.models import (
    build_ref_vaee,
    build_sae,
    build_vaee,
    param_matched_latent_dim,
)
from lcblm.utils.data import NextTokenDataset, typed_dataloader
from lcblm.vaee.models import VAEE, compute_loss

# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    model_name: str
    n_concepts: int
    run_name: str = ""
    train_recon: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    train_l0: list[float] = field(default_factory=list)
    val_l0: list[float] = field(default_factory=list)
    best_val_total: float = float("inf")
    best_val_recon: float = float("inf")
    best_l0: float = float("inf")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _flat_tokens(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return embeddings[mask]


def _early_stop(val_total: list[float], patience: int, min_delta: float) -> bool:
    if patience <= 0 or len(val_total) < patience:
        return False
    recent = val_total[-patience:]
    return recent[0] - min(recent) <= min_delta


# ── Training loops ────────────────────────────────────────────────────────────


def train_vaee(  # noqa: PLR0915
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: VAEEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VAEE, RunResult]:
    input_dim = train_ds.embedding_dimension
    model = build_vaee(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="vaee", n_concepts=cfg.num_embeddings)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc="VAEE", unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
            "ortho": 0.0,
        }
        t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            decoder_weight = (
                model.decoder_first_weight()
                if cfg.topology == "stacked" and cfg.lambda_ortho > 0
                else None
            )
            loss_out = compute_loss(
                target=tokens,
                input=out.recon,
                mu=out.mu,
                alpha=out.alpha,
                prototypes=model.prototypes,
                pi=cfg.pi,
                gamma=cfg.gamma,
                beta=cfg.beta,
                lambda_ent=cfg.lambda_ent,
                lambda_ortho=cfg.lambda_ortho,
                decoder_weight=decoder_weight,
                num_embeddings=model.num_embeddings,
                embedding_size=model.embedding_size,
            )
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            epoch_terms["total"] += loss_out.total_loss.item()
            epoch_terms["recon"] += loss_out.recon_loss.item()
            epoch_terms["cond_kl"] += loss_out.cond_kl_loss.item()
            epoch_terms["sparsity"] += loss_out.sparsity_loss.item()
            epoch_terms["entropy"] += loss_out.entropy_loss.item()
            epoch_terms["ortho"] += loss_out.ortho_loss.item()
            with torch.no_grad():
                t_l0 += (out.c > cfg.l0_threshold).float().sum(dim=1).sum().item()
                t_count += out.c.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
            "ortho": 0.0,
        }
        v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(cfg.device),
                    batch.attention_mask.to(cfg.device),
                )
                out = model(tokens)
                loss_out = compute_loss(
                    target=tokens,
                    input=out.recon,
                    mu=out.mu,
                    alpha=out.alpha,
                    prototypes=model.prototypes,
                    pi=cfg.pi,
                    gamma=cfg.gamma,
                    beta=cfg.beta,
                    lambda_ent=cfg.lambda_ent,
                    lambda_ortho=cfg.lambda_ortho,
                    decoder_weight=(
                        model.decoder_first_weight()
                        if cfg.topology == "stacked" and cfg.lambda_ortho > 0
                        else None
                    ),
                    num_embeddings=model.num_embeddings,
                    embedding_size=model.embedding_size,
                )
                val_terms["total"] += loss_out.total_loss.item()
                val_terms["recon"] += loss_out.recon_loss.item()
                val_terms["cond_kl"] += loss_out.cond_kl_loss.item()
                val_terms["sparsity"] += loss_out.sparsity_loss.item()
                val_terms["entropy"] += loss_out.entropy_loss.item()
                val_terms["ortho"] += loss_out.ortho_loss.item()
                v_l0 += (out.c > cfg.l0_threshold).float().sum(dim=1).sum().item()
                v_count += out.c.shape[0]

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        val_total = val_terms["total"] / n_val
        result.val_recon.append(val_recon)
        result.val_total.append(val_total)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            log_terms = {
                k: v
                for k, v in epoch_terms.items()
                if k != "ortho" or cfg.lambda_ortho > 0
            }
            log_val_terms = {
                k: v
                for k, v in val_terms.items()
                if k != "ortho" or cfg.lambda_ortho > 0
            }
            wandb_run.log(
                {f"train/vaee_{k}": v / n_tr for k, v in log_terms.items()}
                | {f"val/vaee_{k}": v / n_val for k, v in log_val_terms.items()}
                | {
                    "train/vaee_l0": result.train_l0[-1],
                    "val/vaee_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_total < result.best_val_total - cfg.early_stopping_min_delta:
            result.best_val_total = val_total
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "best_val_total": result.best_val_total,
                        "best_val_recon": result.best_val_recon,
                        "best_l0": result.best_l0,
                    },
                )
        elif _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, result


def train_topk_sae(  # noqa: C901, PLR0915
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: TopKSAEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    latent_dim = cfg.latent_dim if cfg.latent_dim > 0 else 4 * input_dim
    model = build_sae(input_dim, latent_dim, TopK(cfg.k), train_ds, cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="topk_sae", n_concepts=latent_dim)
    best_state: dict | None = None

    dead_counts = torch.zeros(latent_dim, dtype=torch.long, device=cfg.device)

    for epoch in trange(cfg.epochs, desc="TopK-SAE", unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {"total": 0.0, "recon": 0.0, "aux": 0.0}
        t_l0 = t_count = 0.0

        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            dead_counts = update_dead_latent_counts(out.latents.detach(), dead_counts)
            dead_mask = dead_counts > cfg.threshold_dead_latent
            recon_loss = F.mse_loss(out.recon, tokens)
            aux_loss = loss_k_aux(model, tokens, out, dead_mask, k_aux=cfg.k_aux)
            loss = loss_top_k(recon_loss, aux_loss, alpha_aux=cfg.alpha_aux)
            optimizer.zero_grad()
            loss.backward()
            if cfg.normalize_decoder:
                model.project_decoder_gradients()
            optimizer.step()
            if cfg.normalize_decoder:
                model.normalize_decoder()
            epoch_terms["total"] += loss.item()
            epoch_terms["recon"] += recon_loss.item()
            epoch_terms["aux"] += cfg.alpha_aux * aux_loss.item()
            with torch.no_grad():
                t_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                t_count += out.latents.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {"total": 0.0, "recon": 0.0, "aux": 0.0}
        v_l0 = v_count = 0.0
        dead_mask_val = dead_counts > cfg.threshold_dead_latent
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(cfg.device),
                    batch.attention_mask.to(cfg.device),
                )
                out = model(tokens)
                recon_loss = F.mse_loss(out.recon, tokens)
                aux_loss = loss_k_aux(
                    model,
                    tokens,
                    out,
                    dead_mask_val,
                    k_aux=cfg.k_aux,
                )
                val_loss = loss_top_k(recon_loss, aux_loss, alpha_aux=cfg.alpha_aux)
                val_terms["total"] += val_loss.item()
                val_terms["recon"] += recon_loss.item()
                val_terms["aux"] += cfg.alpha_aux * aux_loss.item()
                v_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                v_count += out.latents.shape[0]

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        val_total = val_terms["total"] / n_val
        result.val_recon.append(val_recon)
        result.val_total.append(val_total)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/sae_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/sae_{k}": v / n_val for k, v in val_terms.items()}
                | {
                    "train/sae_l0": result.train_l0[-1],
                    "val/sae_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_total < result.best_val_total - cfg.early_stopping_min_delta:
            result.best_val_total = val_total
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "best_val_total": result.best_val_total,
                        "best_val_recon": result.best_val_recon,
                        "best_l0": result.best_l0,
                    },
                )
        elif _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, result


def _train_l1_sae(  # noqa: C901, PLR0913, PLR0915
    model_name: str,
    latent_dim: int,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: SAEConceptConfig | SAEParamConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    model = build_sae(
        input_dim,
        latent_dim,
        nn.ReLU(),
        train_ds,
        cfg.device,
        tied_bias=False,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name=model_name, n_concepts=latent_dim)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc=model_name, unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {"total": 0.0, "recon": 0.0, "l1": 0.0}
        t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            recon_loss = F.mse_loss(out.recon, tokens)
            l1_loss = out.latents.abs().mean()
            loss = recon_loss + cfg.lambda_l1 * l1_loss
            optimizer.zero_grad()
            loss.backward()
            if cfg.normalize_decoder:
                model.project_decoder_gradients()
            optimizer.step()
            if cfg.normalize_decoder:
                model.normalize_decoder()
            epoch_terms["total"] += loss.item()
            epoch_terms["recon"] += recon_loss.item()
            epoch_terms["l1"] += cfg.lambda_l1 * l1_loss.item()
            with torch.no_grad():
                t_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                t_count += out.latents.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {"total": 0.0, "recon": 0.0, "l1": 0.0}
        v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(cfg.device),
                    batch.attention_mask.to(cfg.device),
                )
                out = model(tokens)
                recon_loss = F.mse_loss(out.recon, tokens)
                l1_loss = out.latents.abs().mean()
                val_terms["total"] += (recon_loss + cfg.lambda_l1 * l1_loss).item()
                val_terms["recon"] += recon_loss.item()
                val_terms["l1"] += cfg.lambda_l1 * l1_loss.item()
                v_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                v_count += out.latents.shape[0]

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        val_total = val_terms["total"] / n_val
        result.val_recon.append(val_recon)
        result.val_total.append(val_total)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/sae_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/sae_{k}": v / n_val for k, v in val_terms.items()}
                | {
                    "train/sae_l0": result.train_l0[-1],
                    "val/sae_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_total < result.best_val_total - cfg.early_stopping_min_delta:
            result.best_val_total = val_total
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "best_val_total": result.best_val_total,
                        "best_val_recon": result.best_val_recon,
                        "best_l0": result.best_l0,
                    },
                )
        elif _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, result


def train_sae_concept(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: SAEConceptConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    return _train_l1_sae(
        "sae_concept",
        cfg.vaee_num_embeddings,
        train_ds,
        val_ds,
        cfg,
        wandb_run,
    )


def train_sae_param(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: SAEParamConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    ref_vaee = build_ref_vaee(input_dim, cfg)
    latent_dim = param_matched_latent_dim(ref_vaee, input_dim)
    del ref_vaee
    return _train_l1_sae("sae_param", latent_dim, train_ds, val_ds, cfg, wandb_run)
