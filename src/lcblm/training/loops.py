"""Training loops for VAEE, TopK-SAE, and L1-SAE concept models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

from lcblm.baselines import (
    VQVAE,
    BetaVAE,
    compute_beta_vae_loss,
    compute_vq_vae_loss,
)
from lcblm.eval.metrics import alive_dict_size as _alive_dict_size
from lcblm.sae_utils import SparseAE, TopK
from lcblm.sae_utils.activations import update_dead_latent_counts
from lcblm.sae_utils.losses import loss_k_aux, loss_top_k
from lcblm.training.models import (
    build_beta_vae,
    build_ref_vaee,
    build_sae,
    build_vaee,
    build_vaee_shared_encoder,
    build_vq_vae,
    param_matched_latent_dim,
)
from lcblm.utils.data import typed_dataloader
from lcblm.vaee.models import (
    VAEE,
    VAEESharedEncoder,
    compute_loss,
    compute_loss_shared_encoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import wandb
    from lcblm.training.configs import (
        BetaVAEConfig,
        SAEConceptConfig,
        SAEParamConfig,
        TopKSAEConfig,
        VAEEConfig,
        VAEESharedEncoderConfig,
        VQVAEConfig,
    )
    from lcblm.utils.data import FlatTensorDataset

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
    # Per-model loss-term breakdown by epoch. Each entry of train_losses /
    # val_losses is a dict keyed by the term name (e.g. cond_kl, sparsity,
    # entropy, ortho for VAEE; aux, kl for SAE / β-VAE). One dict per epoch.
    train_losses: list[dict[str, float]] = field(default_factory=list)
    val_losses: list[dict[str, float]] = field(default_factory=list)
    best_val_total: float = float("inf")
    best_val_recon: float = float("inf")
    best_l0: float = float("inf")
    alive_dict_size: int = 0
    matched_fraction: float | None = None
    mean_cosine_sim: float | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _early_stop(val_total: list[float], patience: int, min_delta: float) -> bool:
    if patience <= 0 or len(val_total) < patience:
        return False
    recent = val_total[-patience:]
    return recent[0] - min(recent) <= min_delta


def _vq_one_hot(indices: Tensor, num_codes: int) -> Tensor:
    b = indices.shape[0]
    one_hot = torch.zeros(b, num_codes)
    one_hot.scatter_(1, indices.cpu().unsqueeze(1), 1.0)
    return one_hot


def _compute_alive(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    extract: Callable[[Any], Tensor],
) -> int:
    parts: list[Tensor] = []
    with torch.inference_mode():
        for batch in val_loader:
            out = model(batch.to(device))
            parts.append(extract(out).cpu())
    return _alive_dict_size(torch.cat(parts, dim=0))


def _reset_adam_slice(
    optimizer: optim.Optimizer,
    param: Tensor,
    index: Any,
) -> None:
    """Zero the Adam first/second moments for a slice of ``param``.

    Stops stale momentum from immediately overwriting a freshly reseeded weight
    block. No-op if the parameter has not been stepped yet.
    """
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        buf = state.get(key)
        if buf is not None:
            buf[index] = 0
    if "step" in state:
        # Keep the global step but the bias-correction will treat the reset
        # moments as if fresh; leaving step as-is is the standard choice.
        pass


@torch.no_grad()
def _vaee_concept_directions(model: VAEE) -> Tensor:
    """Input-space direction each concept decodes to (prototype alone, stacked).

    Returns a ``[K, input_dim]`` tensor; row ``i`` is ``decoder(onehot_i ⊗ p_i)``,
    the same per-concept direction the feature-recovery metric matches against the
    ground-truth atoms.
    """
    k, e_dim = model.num_embeddings, model.embedding_size
    dev = model.prototypes.device
    z = torch.zeros(k, k, e_dim, device=dev)
    idx = torch.arange(k, device=dev)
    z[idx, idx] = model.prototypes
    return model._decoder(z.flatten(start_dim=1))  # noqa: SLF001


@torch.no_grad()
def _residual_atom_targets(
    res: Tensor,
    n: int,
    covered: Tensor | None,
    *,
    n_iters: int = 8,
) -> Tensor:
    """Estimate ``n`` uncovered single-atom directions from a residual cloud.

    Reviving a concept toward a single worst-residual *sample* fails: residual norm
    grows with the number of missing atoms in that sample, so the highest-residual
    samples are multi-atom *blends*, not clean atoms. Instead we run spherical
    k-means on the residual directions — each recurring missing atom forms a tight
    cluster (the other atoms in each blend are random and average out), so a
    centroid denoises to that single atom. Already-covered concept directions are
    added as *fixed* centroids so residual mass they explain is absorbed by them and
    the ``n`` free centroids capture only genuinely uncovered directions.

    Binary / positive-coefficient superposition (the high-dim tiers) means atoms
    appear with a single sign, so cosine-argmax assignment is well-posed.

    Returns ``[n, D]`` unit target directions.
    """
    dev = res.device
    rn = F.normalize(res, dim=1)
    res_norm = res.norm(dim=1)
    order = torch.argsort(res_norm, descending=True)
    pool = rn[order[: min(res.shape[0], max(16 * n, 512))]]

    # Farthest-point init over the high-residual pool, avoiding covered directions.
    if covered is not None and covered.numel():
        max_sim = (pool @ covered.T).amax(dim=1)
    else:
        max_sim = torch.zeros(pool.shape[0], device=dev)
    init: list[int] = []
    for _ in range(n):
        j = int(torch.argmin(max_sim).item())
        init.append(j)
        max_sim = torch.maximum(max_sim, pool @ pool[j])
    centroids = pool[init].clone()  # [n, D] free centroids

    fixed = covered if (covered is not None and covered.numel()) else None
    for _ in range(n_iters):
        allc = centroids if fixed is None else torch.cat([centroids, fixed], dim=0)
        assign = (rn @ allc.T).argmax(dim=1)  # [N]
        for c in range(n):
            m = assign == c
            if bool(m.any()):
                centroids[c] = F.normalize(res[m].mean(dim=0), dim=0)
    return centroids


@torch.no_grad()
def _resample_dead_vaee_concepts(  # noqa: PLR0913
    model: VAEE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: VAEEConfig,
    device: torch.device,
    *,
    max_resample: int,
) -> int:
    """Reinitialise dead VAEE concepts toward under-reconstructed data directions.

    A concept is *dead* if its gate fires (``c > l0_threshold``) on fewer than
    ``cfg.resample_dead_frac`` of the samples in ``loader``. Each dead concept is
    reseeded as a consistent (encoder-rows, prototype, decoder-block) triple that
    fires on, and decodes to, the residual direction of a worst-reconstructed
    sample. Adam moments for the touched slices are reset.

    Returns the number of concepts resampled (0 if the encoder/decoder layout is
    unsupported or nothing is dead).
    """
    enc = model.encoder_concept_linear()
    dec = model.decoder_concept_linear()
    if enc is None or dec is None or model.topology != "stacked":
        return 0

    k, e_dim = model.num_embeddings, model.embedding_size
    threshold = cfg.l0_threshold

    fire_counts = torch.zeros(k, device=device)
    n_seen = 0
    residuals: list[Tensor] = []
    was_training = model.training
    model.eval()
    for batch in typed_dataloader(loader):
        x = batch.to(device)
        out = model(x)
        fire_counts += (out.c > threshold).float().sum(dim=0)
        n_seen += x.shape[0]
        residuals.append((x - out.recon).detach())
    if was_training:
        model.train()

    if n_seen == 0:
        return 0
    fire_rate = fire_counts / n_seen
    dead = (fire_rate < cfg.resample_dead_frac).nonzero(as_tuple=True)[0].tolist()
    if not dead:
        return 0
    if max_resample > 0:
        dead = dead[:max_resample]

    res = torch.cat(residuals, dim=0)  # [N, D]
    n = min(len(dead), res.shape[0])
    dead = dead[:n]

    # Reseed toward *uncovered single-atom* directions recovered by clustering the
    # residual cloud (see _residual_atom_targets), with the currently-alive concepts
    # supplied as fixed centroids so revived concepts target genuinely missing atoms.
    alive_mask = fire_rate >= cfg.resample_dead_frac
    if bool(alive_mask.any()):
        covered = F.normalize(
            _vaee_concept_directions(model)[alive_mask], dim=1
        )  # [A, D]
    else:
        covered = None
    targets = _residual_atom_targets(res, n, covered)  # [n, D]

    w_enc = enc.weight.data
    b_enc = enc.bias.data if enc.bias is not None else None
    w_dec = dec.weight.data
    for slot, ci in enumerate(dead):
        r = targets[slot]  # [D]
        p = torch.randn(e_dim, device=device)  # match prototype init scale
        rows = slice(ci * e_dim, (ci + 1) * e_dim)
        # encoder rows: Enc_i @ r = p  ⇒  pre-activation ≈ p for x ≈ r (gate fires)
        w_enc[rows, :] = torch.outer(p, r)
        if b_enc is not None:
            b_enc[rows] = 0.0
        # decoder block: W_i @ p = r  ⇒  concept decodes to the residual direction
        w_dec[:, rows] = torch.outer(r, p) / p.dot(p).clamp(min=1e-8)
        model.prototypes.data[ci] = p

        _reset_adam_slice(optimizer, enc.weight, rows)
        if enc.bias is not None:
            _reset_adam_slice(optimizer, enc.bias, rows)
        _reset_adam_slice(optimizer, dec.weight, (slice(None), rows))
        _reset_adam_slice(optimizer, model.prototypes, ci)

    return len(dead)


# ── Training loops ────────────────────────────────────────────────────────────


def train_vaee(  # noqa: PLR0915
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: VAEEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VAEE, RunResult]:
    input_dim = train_ds.input_dim
    model = build_vaee(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="vaee", n_concepts=cfg.num_embeddings)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc="VAEE", unit="epoch"):
        lambda_ent_eff = (
            cfg.lambda_ent * min(1.0, (epoch + 1) / cfg.lambda_ent_warmup_epochs)
            if cfg.lambda_ent_warmup_epochs > 0
            else cfg.lambda_ent
        )
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
            tokens = batch.to(cfg.device)
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
                lambda_ent=lambda_ent_eff,
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
                tokens = batch.to(cfg.device)
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
                    lambda_ent=lambda_ent_eff,
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

        in_warmup = (epoch + 1) <= cfg.lambda_ent_warmup_epochs
        if (
            not in_warmup
            and val_total < result.best_val_total - cfg.early_stopping_min_delta
        ):
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
        elif not in_warmup and _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

        if (
            cfg.resample_dead
            and (epoch + 1) % cfg.resample_every == 0
            and (epoch + 1) <= cfg.epochs * cfg.resample_stop_frac
        ):
            n_res = _resample_dead_vaee_concepts(
                model,
                train_loader,
                optimizer,
                cfg,
                cfg.device,
                max_resample=cfg.resample_max_per_step,
            )
            if n_res:
                print(f"   Resampled {n_res} dead concept(s) at epoch {epoch + 1}")
                if wandb_run is not None:
                    wandb_run.log({"train/vaee_resampled": n_res}, step=epoch + 1)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    threshold = cfg.l0_threshold
    result.alive_dict_size = _compute_alive(
        model,
        val_loader,
        cfg.device,
        lambda out: (out.c > threshold).float(),
    )
    return model, result


def train_vaee_shared_encoder(  # noqa: PLR0915
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: VAEESharedEncoderConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VAEESharedEncoder, RunResult]:
    input_dim = train_ds.input_dim
    model = build_vaee_shared_encoder(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="vaee_shared_encoder", n_concepts=cfg.num_embeddings)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc="VAEESharedEncoder", unit="epoch"):
        lambda_ent_eff = (
            cfg.lambda_ent * min(1.0, (epoch + 1) / cfg.lambda_ent_warmup_epochs)
            if cfg.lambda_ent_warmup_epochs > 0
            else cfg.lambda_ent
        )
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
            tokens = batch.to(cfg.device)
            out = model(tokens)
            decoder_weight = (
                model.decoder_first_weight()
                if cfg.topology == "stacked" and cfg.lambda_ortho > 0
                else None
            )
            loss_out = compute_loss_shared_encoder(
                target=tokens,
                input=out.recon,
                alpha=out.alpha,
                pi=cfg.pi,
                beta=cfg.beta,
                lambda_ent=lambda_ent_eff,
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
                tokens = batch.to(cfg.device)
                out = model(tokens)
                loss_out = compute_loss_shared_encoder(
                    target=tokens,
                    input=out.recon,
                    alpha=out.alpha,
                    pi=cfg.pi,
                    beta=cfg.beta,
                    lambda_ent=lambda_ent_eff,
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
                {f"train/vaee_se_{k}": v / n_tr for k, v in log_terms.items()}
                | {f"val/vaee_se_{k}": v / n_val for k, v in log_val_terms.items()}
                | {
                    "train/vaee_se_l0": result.train_l0[-1],
                    "val/vaee_se_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        in_warmup = (epoch + 1) <= cfg.lambda_ent_warmup_epochs
        if (
            not in_warmup
            and val_total < result.best_val_total - cfg.early_stopping_min_delta
        ):
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
        elif not in_warmup and _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    threshold = cfg.l0_threshold
    result.alive_dict_size = _compute_alive(
        model, val_loader, cfg.device, lambda out: (out.c > threshold).float()
    )
    return model, result


def train_topk_sae(  # noqa: C901, PLR0915
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: TopKSAEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.input_dim
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
            tokens = batch.to(cfg.device)
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
                tokens = batch.to(cfg.device)
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
    result.alive_dict_size = _compute_alive(
        model,
        val_loader,
        cfg.device,
        lambda out: out.latents,
    )
    return model, result


def _train_l1_sae(  # noqa: C901, PLR0913, PLR0915
    model_name: str,
    latent_dim: int,
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: SAEConceptConfig | SAEParamConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.input_dim
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
            tokens = batch.to(cfg.device)
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
                tokens = batch.to(cfg.device)
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
    result.alive_dict_size = _compute_alive(
        model,
        val_loader,
        cfg.device,
        lambda out: out.latents,
    )
    return model, result


def train_sae_concept(
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: SAEConceptConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    return _train_l1_sae(
        "sae_concept",
        cfg.latent_dim,
        train_ds,
        val_ds,
        cfg,
        wandb_run,
    )


def train_sae_param(
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: SAEParamConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.input_dim
    ref_vaee = build_ref_vaee(input_dim, cfg)
    latent_dim = param_matched_latent_dim(ref_vaee, input_dim)
    del ref_vaee
    return _train_l1_sae("sae_param", latent_dim, train_ds, val_ds, cfg, wandb_run)


def train_vq_vae(  # noqa: C901, PLR0915
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: VQVAEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VQVAE, RunResult]:
    input_dim = train_ds.input_dim
    model = build_vq_vae(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="vq_vae", n_concepts=cfg.num_codes)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc="VQ-VAE", unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "codebook": 0.0,
            "commitment": 0.0,
        }
        t_codes_used: set[int] = set()
        for batch in typed_dataloader(train_loader):
            tokens = batch.to(cfg.device)
            out = model(tokens)
            loss_out = compute_vq_vae_loss(
                target=tokens,
                out=out,
                commitment_weight=cfg.commitment_weight,
            )
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            epoch_terms["total"] += loss_out.total_loss.item()
            epoch_terms["recon"] += loss_out.recon_loss.item()
            epoch_terms["codebook"] += loss_out.codebook_loss.item()
            epoch_terms["commitment"] += (
                cfg.commitment_weight * loss_out.commitment_loss.item()
            )
            with torch.no_grad():
                t_codes_used.update(out.indices.unique().cpu().tolist())

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        # Per-sample L0 is 1 by construction for hard VQ.
        result.train_l0.append(1.0)

        if cfg.reset_dead_codes:
            dead = [c for c in range(model.num_codes) if c not in t_codes_used]
            if dead:
                sample = next(iter(train_loader)).to(cfg.device)
                with torch.no_grad():
                    z_e = model.encode(sample)
                    perm = torch.randperm(z_e.shape[0], device=cfg.device)
                    for i, code_idx in enumerate(dead):
                        model.codebook.data[code_idx] = z_e[perm[i % perm.shape[0]]]

        model.eval()
        val_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "codebook": 0.0,
            "commitment": 0.0,
        }
        v_codes_used: set[int] = set()
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = batch.to(cfg.device)
                out = model(tokens)
                loss_out = compute_vq_vae_loss(
                    target=tokens,
                    out=out,
                    commitment_weight=cfg.commitment_weight,
                )
                val_terms["total"] += loss_out.total_loss.item()
                val_terms["recon"] += loss_out.recon_loss.item()
                val_terms["codebook"] += loss_out.codebook_loss.item()
                val_terms["commitment"] += (
                    cfg.commitment_weight * loss_out.commitment_loss.item()
                )
                v_codes_used.update(out.indices.unique().cpu().tolist())

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        val_total = val_terms["total"] / n_val
        result.val_recon.append(val_recon)
        result.val_total.append(val_total)
        result.val_l0.append(1.0)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/vq_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/vq_{k}": v / n_val for k, v in val_terms.items()}
                | {
                    "train/vq_codes_used": len(t_codes_used),
                    "val/vq_codes_used": len(v_codes_used),
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
    num_codes = cfg.num_codes
    result.alive_dict_size = _compute_alive(
        model,
        val_loader,
        cfg.device,
        lambda out: _vq_one_hot(out.indices, num_codes),
    )
    return model, result


def train_beta_vae(  # noqa: PLR0915
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    cfg: BetaVAEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[BetaVAE, RunResult]:
    input_dim = train_ds.input_dim
    model = build_beta_vae(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="beta_vae", n_concepts=cfg.latent_dim)
    best_state: dict | None = None

    for epoch in trange(cfg.epochs, desc="β-VAE", unit="epoch"):
        beta_eff = (
            cfg.beta * min(1.0, (epoch + 1) / cfg.kl_warmup_epochs)
            if cfg.kl_warmup_epochs > 0
            else cfg.beta
        )
        model.train()
        epoch_terms: dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = batch.to(cfg.device)
            out = model(tokens)
            loss_out = compute_beta_vae_loss(target=tokens, out=out, beta=beta_eff)
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            epoch_terms["total"] += loss_out.total_loss.item()
            epoch_terms["recon"] += loss_out.recon_loss.item()
            epoch_terms["kl"] += beta_eff * loss_out.kl_loss.item()
            with torch.no_grad():
                active = (out.mu.abs() > cfg.l0_threshold).float()
                t_l0 += active.sum(dim=1).sum().item()
                t_count += out.mu.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "kl_raw": 0.0,
        }
        v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = batch.to(cfg.device)
                out = model(tokens)
                loss_out = compute_beta_vae_loss(
                    target=tokens,
                    out=out,
                    beta=beta_eff,
                )
                val_terms["total"] += loss_out.total_loss.item()
                val_terms["recon"] += loss_out.recon_loss.item()
                val_terms["kl"] += beta_eff * loss_out.kl_loss.item()
                val_terms["kl_raw"] += loss_out.kl_loss.item()
                active = (out.mu.abs() > cfg.l0_threshold).float()
                v_l0 += active.sum(dim=1).sum().item()
                v_count += out.mu.shape[0]

        n_val = len(val_loader)
        val_recon = val_terms["recon"] / n_val
        # Model selection uses the fixed-β ELBO, not the warmup-weighted total:
        # while β_eff ramps up the live total is non-comparable across epochs
        # (it spuriously favours epoch 1). recon + β·KL_raw stays comparable.
        val_total = val_recon + cfg.beta * (val_terms["kl_raw"] / n_val)
        result.val_recon.append(val_recon)
        result.val_total.append(val_total)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/bvae_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/bvae_{k}": v / n_val for k, v in val_terms.items()}
                | {
                    "train/bvae_l0": result.train_l0[-1],
                    "val/bvae_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        in_warmup = (epoch + 1) <= cfg.kl_warmup_epochs
        if (
            not in_warmup
            and val_total < result.best_val_total - cfg.early_stopping_min_delta
        ):
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
        elif not in_warmup and _early_stop(
            result.val_total,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    l0_threshold = cfg.l0_threshold
    result.alive_dict_size = _compute_alive(
        model,
        val_loader,
        cfg.device,
        lambda out: (out.mu.abs() > l0_threshold).float(),
    )
    return model, result
